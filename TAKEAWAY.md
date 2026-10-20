# torch.compile

## eval_frame_cpp 内部机制
- 该模块处于 CPython 执行器和 Py侧 TorchDynamo 之间, 拦截 frame, 查询缓存, 执行后续代码
- 解决四个问题:
    1. 当前 frame 要不要交给 Dynamo;
    2. 能不能复用已有的编译结果;
    3. 未命中时, 调用哪一个callback;
    4. 返回后执行原字节码还是修改后的字节码.

### set_eval_frame
- Torch 对上暴露 set_eval_frame 包装器, 调用 set_eval_frame 时, 根据传入的 callback 类型决定 compile 执行模式;
- 线程级的hook: 在 set_eval_frame 内, 先通过 eval_frame_callback_get() 获取 thread_local 的 callback, 根据具体值, 决定是否替换这个 callback
- 解释器级的hook: 除了替换 callback, 还需要在 frame 执行之前, 替换 CPython 的 evaluator (enable_eval_frame_shim), 之后Cpython 执行每个 frame 会先进入 shim 函数, 然后重新读tls callback, 决定如何执行这个frame
    - 为啥要重新读tls callback: 
        - 因为callback是线程级的, 而shim是解释器级; 如果 thread A callback = callback 开启 Dynamo 之后, 所有线程的evaluator 都会变成 shim, 但是此时 thread B callback = None, 依然可以走 default evaluator
        - Dynamo 做 guard lookup 前会把callback设置成None, 防止递归调用; 需要支持递归frame时, 会有其他策略

## Dynamo Cache Miss的处理路径

### default
- cache hit 时执行缓存, cache miss时执行 callback, 重新生成 dynamo cache

### run_only (eager_on_recompile)
- cache hit 时执行缓存, cache miss时执行default frame, 走eager模式

### fail_on_recompile
- `warmup -> torch.compiler.set_stance("fail_on_recompile")`
- cache hit 时执行缓存, cache miss 时执行 fail_callback, 产生 RuntimeError

### skip_guard_eval_unsafe
- `torch.compiler.set_stance(skip_guard_eval_unsafe= True)`
- 在充分warmup之后, 开启这个来减少guard检查的开销(不执行全量的检查), 如果没有充分warmup, 可能会执行错误的cache code
```cpp
void lookup() {
if (is_skip_guard_eval_unsafe) {
    valid = torch::dynamo::run_root_guard_manager(
        cache_entry.diff_guard_root_mgr, f_locals);
} else {
    valid = torch::dynamo::run_root_guard_manager(
        cache_entry.root_mgr, f_locals);
}
}
```

### Stance 模式汇总

| stance | 目标 | `_callback_from_stance` 返回值 | cache hit | cache miss / 首次调用 |
| --- | --- | --- | --- | --- |
| `default` | 正常使用 `torch.compile` | 原 callback；指定 `force_backend` 时返回对应 backend 的包装 callback | 执行缓存 | 调用 callback 编译并写入缓存 |
| `force_eager` | 完全关闭 Dynamo | `None` | 不查询缓存 | 直接执行原始字节码 |
| `eager_on_recompile` | 只使用已有缓存，禁止产生新的编译结果 | `False`，即 run-only | 执行缓存 | 直接执行原始字节码 |
| `fail_on_recompile` | 检查预热是否完整，禁止出现新的输入变体 | `fail_callback` | 执行缓存 | `fail_callback` 抛出 `RuntimeError` |
| `eager_then_compile` | 先观察两次输入，再按动态性编译 | delayed callback | 已有匹配缓存时直接执行 | 第一次 eager 并记录输入；第二次比较两份输入，用原 backend 编译 |
| `aot_eager_then_compile` | 第一次保留 AOTAutograd 的分图和重计算能力，之后再使用原 backend | delayed callback；第一次内部调用 `aot_eager` | 已有匹配缓存时直接执行 | 第一次用 `aot_eager` 并记录输入；后续 miss 时比较输入，用原 backend 编译 |

`skip_guard_eval_unsafe` 是附加开关，不是 stance。它不替换 callback，只把 cache lookup 使用的 guard 从 `root_mgr` 切换为 `diff_guard_root_mgr`。

## Guard 类型和校验方式

### Guard Tree

- `run_root_guard_manager()` 没有一套固定的检查项。Dynamo trace 时只为实际依赖的对象建立 guard, 运行时执行对应 `CacheEntry` 保存的 guard tree。
- Guard tree 由两类节点组成:
    - `GuardAccessor`: 从父对象取出待检查的值, 例如 `f_locals["x"]`、`obj.attr`、`dict[key]`、`tensor.size(0)`。
    - `LeafGuard`: 对取出的值做判断, 例如类型、对象身份、Tensor 元数据或 shape 约束。
- 执行顺序:
    1. root leaf guards, 主要检查全局运行时状态;
    2. accessors 取出 locals、globals、属性和容器元素, 递归执行子 GuardManager;
    3. epilogue guards, 执行无法直接放到对象节点上的 Python/SymPy 表达式。
- 任意 guard 返回 `false` 都会立即停止当前 `CacheEntry` 的匹配。Accessor 子树会按历史失败次数调整顺序, 让更容易失败的检查优先执行。

### 基础对象 Guard

| Guard | 校验手段 | 典型用途 |
| --- | --- | --- |
| `TYPE_MATCH` | 比较 `Py_TYPE(value)` 指针 | 参数类型、容器类型、Tensor 子类 |
| `ID_MATCH` | 比较对象指针, 等价于 `value is expected` | 函数、类、module、特定常量对象 |
| `EQUALS_MATCH` | 先检查类型, 再调用 `PyObject_RichCompareBool(..., Py_EQ)` | int、str、tuple、slice、`torch.Size` 等常量 |
| `NONE_MATCH` | `value == Py_None` | Optional 参数特化为 `None` |
| `TRUE_MATCH` / `FALSE_MATCH` | 比较 Python bool 单例 | 依赖布尔分支的代码 |
| `NOT_NONE` | `value != Py_None` | weakref 结果、Optional Tensor |
| `FLOAT_IS_NAN` / `COMPLEX_IS_NAN` | 检查精确数值类型并调用 `isnan` | 避免 `nan != nan` 导致 `EQUALS_MATCH` 永远失败 |
| `NO_HASATTR` | `PyObject_HasAttr(...) == 0` | trace 时依赖某个属性不存在 |
| `LAMBDA_GUARD` | 调用 Python guard 函数并取其 truth value | 尚未下沉到 C++ 的特殊检查 |

### 容器和迭代器 Guard

| Guard | 校验手段 |
| --- | --- |
| `LENGTH_CHECK` | `PySequence_Length(value) == expected` |
| `DICT_LENGTH` | 确认是 dict, 再比较 `PyDict_Size` |
| `DICT_VERSION` | 比较 CPython dict version tag |
| `DICT_CONTAINS` | `PyDict_Contains` 检查 key 存在或不存在 |
| `SET_CONTAINS` | `PySet_Contains` 检查元素存在或不存在 |
| `MAPPING_KEYS_MATCH` | 获取当前 keys, 与 trace 时保存的 keys 列表比较 |
| `TUPLE_ITERATOR_LEN` | 比较 tuple iterator 的剩余元素数 |
| `RANGE_ITERATOR_MATCH` | 比较迭代器归一化后的 `start`、`stop`、`step` |

- 普通 dict 还可能使用 `DictGuardManager`: 先检查精确类型和长度, 再按 `PyDict_Next` 的顺序检查 trace 实际依赖的 key/value, 不会无条件递归整个 dict。
- dict tag 没变化且子树满足 tag-safe 条件时, 可以跳过整棵不可变子树的逐项检查。

### Tensor Guard

`TENSOR_MATCH` 是 Tensor 输入最主要的检查, 依次校验:

1. Python Tensor 类型或 Tensor 子类;
2. 应用 TLS 修正后的 `DispatchKeySet`;
3. `dtype`;
4. device index;
5. `requires_grad`;
6. rank;
7. 静态维度的 size;
8. 静态维度的 stride。

动态维度在 `TensorCheck` 中保存为 `nullopt`, `TENSOR_MATCH` 不比较该维的具体值。动态 shape 是否可复用, 继续由 shape guards 判断。

相关 Tensor guard:

| Guard | 校验手段 |
| --- | --- |
| `DISPATCH_KEY_SET_MATCH` | 比较应用当前 TLS include/exclude 后的 dispatch key raw representation |
| `DYNAMIC_INDICES` | 检查本次请求的 dynamic 维度与已编译维度是否兼容 |
| `SYMBOLIC_SHAPE_GUARD` | 收集多个 size、stride 或 scalar, 调用 ShapeEnv 生成的 C guard 函数 |

`SYMBOLIC_SHAPE_GUARD` 检查的是维度之间的关系, 例如 `x.size(0) == y.size(1)` 或 `x.size(0) >= 2`, 不只是单个 Tensor 的 shape。

### 运行时状态 Guard

| Guard | 校验内容 |
| --- | --- |
| `GLOBAL_STATE` | grad mode、autocast、`__torch_function__` 状态、deterministic algorithms、TF32、FP16/BF16 reduction、CPU 线程数、默认 dtype |
| `TORCH_FUNCTION_MODE_STACK` | mode stack 长度以及每层 mode 的类型 |
| `DEFAULT_DEVICE` | `torch.utils._device.CURRENT_DEVICE` |
| `DUAL_LEVEL_MATCH` | forward AD 当前 level |

这些 guard 一般挂在 root 上, 在读取函数参数之前执行。全局状态发生变化时, 旧图不能直接复用。

### 多对象关系 Guard

| Guard | 校验手段 |
| --- | --- |
| `OBJECT_ALIASING` | 两个 source 取出的对象必须仍是同一个对象 |
| `NO_TENSOR_ALIASING` | 一组原本不同的 Tensor 不能出现重复对象指针 |
| `STORAGE_OVERLAPPING` | 输入 Tensor 之间的 storage overlap 关系保持不变 |
| `SYMBOLIC_SHAPE_GUARD` | 收集多个 source 的值后统一判断 shape 关系 |

- 关系 guard 会共享安装到多个 GuardManager 节点。同一轮 guard tree 遍历中, 前几个相关节点负责收集值, 收齐后再做比较。
- 这些 guard 内部有临时状态, 所以每次整棵树执行结束, 无论成功还是失败, `RootGuardManager` 都会调用 `reset_state()`。

### 常用 Accessor

| Accessor | 取值方式 |
| --- | --- |
| `FrameLocalsGuardAccessor` | 按 localsplus 索引读取函数局部变量, 避免构造完整 `f_locals` dict |
| `GlobalsGuardAccessor` | 进入函数的 globals dict |
| `GetAttrGuardAccessor` | `PyObject_GetAttr` |
| `GenericGetAttrGuardAccessor` | `PyObject_GenericGetAttr` |
| `GetItemGuardAccessor` | `PyObject_GetItem` |
| `DictGetItemGuardAccessor` | 使用 dict C API 按 key 取值 |
| `ListGetItemGuardAccessor` / `TupleGetItemGuardAccessor` | 按固定下标取值 |
| `TensorPropertyGuardAccessor` | 读取 Tensor 的 size、stride 或 storage offset |

Accessor 取值失败也视为 guard failure。例如 trace 时使用了 `module.training`, 运行时该属性消失, 会在 accessor 阶段失败, 不会继续执行子节点。

## compile 问题排查

### 先看是否切图或重编译

```bash
TORCH_LOGS="graph_breaks,recompiles" python run.py
```

- `graph_breaks`: 查看 graph break 的位置和原因。
- `recompiles`: 查看哪条 guard 失败导致重新编译。
- shape 变化较多时, 再加上 `dynamic`。
- 想看实际生成的 guard tree, 使用 `TORCH_LOGS="guards"`。

### 按 backend 定位问题层

```python
torch.compile(fn, backend="eager")      # 只经过 Dynamo
torch.compile(fn, backend="aot_eager")  # Dynamo + AOTAutograd
torch.compile(fn, backend="inductor")   # 完整流程
```

前一种 backend 正常, 后一种失败, 问题通常在后一种新增的编译层。

### 查看中间结果

| 需要确认的内容 | 日志 |
| --- | --- |
| Dynamo 捕获的 FX 图 | `graph_code` |
| AOTAutograd 前向/反向图 | `aot_graphs` |
| Inductor 融合和调度 | `fusion,schedule` |
| Inductor 生成的代码 | `output_code,kernel_code` |

### 常见问题

- 大量 graph break: 检查数据相关分支、`.item()`、打印和不支持的 Python 调用。
- 频繁重编译: 检查 shape、Python scalar、全局变量、`module.training` 和容器结构是否变化。
- 首次调用慢: 区分正常冷启动和重复编译, 可用 `torch._dynamo.utils.compile_times()` 查看各阶段耗时。
- 结果错误或崩溃: 先用不同 backend 缩小范围, 再为对应层开启日志。
- 大模型或分布式任务不方便缩小复现时, 使用 `TORCH_TRACE=/tmp/torch_trace` 收集完整编译过程。
