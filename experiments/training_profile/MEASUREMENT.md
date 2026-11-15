# 测量与归因合同

## 计算保持什么

从已安装 Brax 读取函数源码，仅插入 `with jax.named_scope(...)`。剥除新增 scope 后的 AST 必须完全相同；默认参数、globals、函数签名、表达式、PRNG顺序和调用次数保持原样，退出上下文恢复原函数。不修改安装文件，不用 Python 每步回调计时，也不拆开 JIT 强制同步。

阶段包括 rollout、batch_prepare、normalizer、shuffle、gradient、optimizer 和 bookkeeping；loss 另标 actor_forward、critic_forward、gae、loss_terms。任务适配可给 physics、control、reward、obs 及网络调用添加元数据。梯度前向/反向常被融合，只有不可混淆的独立证据才可进一步细分。

AST相等不能保证独立编译后的浮点轨迹逐位一致。记录相同起始参数、首epoch/最终参数差异、同一 executable 同输入重复结果；不得在发现误差后调宽已声明阈值来宣称通过。

## 计什么时间

编译前输入等待、tracing/lowering、XLA编译分别用单调时钟记录。稳态 epoch 记录完整调用、返回前耗时、返回后的完成等待。返回前耗时可能已经包含设备工作或内部同步，不能称为纯 CPU 提交开销。

只在 epoch 完成边界等待，设备 kernel 时间来自 CUPTI trace。主机的 `StepTraceAnnotation` 覆盖一次完整 epoch，不能与包含在其中的设备时间再次相加。GPU事件按 trace 的 process/thread metadata 识别，只统计设备 stream 的完整事件；CUDA API、CPU Executor 和汇总 stream 不视为设备执行。

跨 stream 取区间并集。独占类别与跨类别重叠可以相加；各类别自身的 busy 并集可能重叠，不能直接求和。主机窗口减去设备 busy 的差值只能叫空隙/未追踪部分，不能直接归因为 Python、CPU计算或某一种同步。

## 元数据与缓存

JAX持久缓存默认可忽略元数据，因此缓存命中的 executable 可能保留旧标签。诊断进程设置 `jax_compilation_cache_include_metadata_in_key=True`；生产环境不因此改动。`Lowered.compiler_ir('hlo').as_hlo_text()` 默认文本不显示元数据，检查标签应使用 `Lowered.as_text(debug_info=True)`，编译后的 `Compiled.as_text()`另存档。

直接 trace 标签是首选。缓存或融合导致标签缺失时，离线分析可用同一工作负载的 compiled HLO 与带标签的 lowered MLIR 建立有来源的精确关联：记录匹配、歧义、混合和未知。只有明确的对应关系才能归因；不能用 kernel 名字的直觉猜测阶段，也不能把 XLA cost estimate 当真实时间。

`hlo_op=command_buffer_N`代表分组，不能当作单一HLO。此时仅允许在同模块内使用固定符号规范化后的kernel关联；多候选保留分类共识或mixed，不作模糊/去数字匹配。分析器另列原始函数路径的subsystem归因，它与`ppo.*`阶段归因分别计账。参考HLO来自另一次编译时，模块名/符号匹配仍不足以证明执行二进制相同；输出只能作为有此限制的热点证据。

## 能力与结果

搬运汇总另列`transfers.device`与`transfers.host_api`，统计完整窗口内明确拷贝的方向、事件数、字节和区间并集；跨窗口事件整条排除并计数，不按时间比例估字节。未知字节/未观测方向保留null及已知字节下界。`kernel_details`标记的计算kernel不按名字判作PCIe。`host_sync_api`只记录明确的CPU同步/等待API；未捕获时unsupported，不推断零阻塞。

运行适配目前限定单 GPU、当前 Tenny 任务及已支持的 Brax 结构。阶段标记和 trace 分析可复用于其他任务；新任务需要明确适配，而非更改通用解析规则来凑数字。未识别结构/设备事件、归因不足或数值门槛失败都应显式记录。

profiler 开销用原版、标记但未采集、标记并采集的对应 epoch 对照。显著扰动时，设备热点可用于确定优先级，但其墙钟和阶段百分比不能直接回填历史训练的84%。本机实际覆盖、失败与结论以[RESULTS.md](RESULTS.md)为准。

计时依据：[JAX异步调度](https://docs.jax.dev/en/latest/async_dispatch.html)、[JAX profiler](https://docs.jax.dev/en/latest/profiling.html)、[缓存元数据选项](https://docs.jax.dev/en/latest/config_options.html#compilation-cache-include-metadata-in-key)。
