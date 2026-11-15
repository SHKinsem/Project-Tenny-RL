# 采样/PPO诊断结果（2026-11-15）

本轮把优化优先级缩小到了 **MJX物理计算，尤其约束求解器**。已有设备事件中，约73.24%的GPU忙时可沿参考HLO追溯到物理求解/前向路径。这个结果是热点线索：采集扰动明显，参考HLO来自另一次相同工作负载的编译，尚不能将其当作生产采样/PPO那84.1%墙钟的精确拆分。

后续已完成同一executable HLO的128/256/512环境对照，吞吐/显存/搬运结果见[并行规模实测](SCALING-RESULTS.md)。本页保留最初阶段标记诊断的失败与边界，不由后续不同验收覆盖。

阶段标记、CPU离线分析、有截止时间的隔离诊断入口和回归已落盘。生产代码、安装的Brax、最新模型及导出均未改变；没有部署优化、重启控制台或追加正式训练。

## 工作负载与计时口径

输入是原生评估接入后保存的run `20261115T053612.022810Z_47e4b501`，checkpoint `000001171456`。源指纹 `4ca2bc7387038679a16eaedde390b4f4db7b6e678007d83fba268ccc9137960a`，checkpoint摘要 `2b4e5d50e7318fe17df788e159a64718008c91594f4d4e10111d27624e3c24cd`。上轮完整训练中采样/PPO含首段JIT占84.10%，依据见[训练对照](../native_eval/TRAINING-INTEGRATION-RESULTS.md)。

沿用128环境、batch_size=128、1个minibatch、每batch更新1次、unroll_length=16及原随机策略、观测归一化、奖励和学习率。每epoch为8192环境控制步；控制100Hz、物理400Hz。诊断只改变总epoch数量，并关闭独立评估/模型写出；参数更新留在隔离进程内。

本机环境是RTX 2080 SUPER、JAX 0.9.2、MuJoCo 3.12.0。复用生产的环境准备函数和实际缓存目录；缓存并不位于torque200自身目录。模型为nq=13/nv=12、10个body、15个geom、2个equality，使用Newton、IMPLICITFAST和AUTO Jacobian。求解器上限100、线搜索上限50、tolerance=1e-8；这些是上限，**尚未测出实际迭代分布**。XML默认步长0.004被任务配置覆盖为0.0025。

## 已采集的设备证据

首次采集中3个完整epoch含24,576环境控制步。CUPTI记录630,604个设备事件；跨stream忙时并集1.059780秒。以下使用互斥时间分区，分母始终是这个设备忙时，不是生产墙钟：

| HLO来源类别 | 独占设备时间 | 忙时占比 |
| --- | ---: | ---: |
| 物理约束求解 `step/forward/solve` | 0.521482秒 | 49.21% |
| 其他物理前向 `step/forward` | 0.254733秒 | 24.04% |
| 来源未知 | 0.281335秒 | 26.55% |
| 来源混合 | 0.002222秒 | 0.21% |
| 跨类别重叠 | 0.000007秒 | 0.00068% |

最频繁的归约、选择和乘法融合kernel，每类约7.6万次，平均单次约1–2.5微秒。沿参考HLO可看到约束求解器内的嵌套while/线搜索来源；大量短kernel和循环控制是值得继续验证的方向。事件次数本身不能证明Newton或线搜索每次都达到上限。

参考映射使用第一次实际trace与第三次同冻结工作负载的**原版compiled HLO**；第三次带标签的lowered MLIR仅用于尝试恢复阶段标签。分析器检查模块名，对58,413个事件使用精确hlo_op，对555,362个command_buffer事件使用固定CUDA符号规范化后的分类一致映射，其余16,829个事件未映射。不删除数字后缀、不作模糊匹配；多候选跨类别保留mixed。分类一致不代表指令唯一，也没有证明参考HLO与第一次执行的二进制完全相同。因此表中结果应作为优先级证据，正式占比仍需同一executable采集复核。

## 未通过的验收及其影响

- 命名阶段恢复覆盖仅0.0488%。不能给出可靠的rollout、GAE、critic、反向传播或optimizer比例，也不能把未知部分都算作PPO更新。
- 首次标记版未采集的对应epoch时间比中位为1.162，样本仅2对；采集版对应epoch时间比中位为5.973。显著扰动使采集墙钟不适合估计生产收益。
- 3个主机epoch窗口合计12.872953秒，设备忙时1.059780秒。其差值包含采集成本、空隙和未追踪工作，不能称为Python、GIL或CPU瓶颈。
- 原版/标记版起始参数差为0；首epoch参数最大差5.9187e-5，超过事先声明的1e-5门槛，严格验收失败。另一次同一原始executable、同输入和PRNG的重复也有5.8025e-5差异。这说明自然不确定性需要计量，不能据此证明标记无影响或将失败改为通过。首次最终参数差1.4389e-4，物理状态最大差3.7257，均保留在原记录中。
- 标记变换剥除新增scope后的AST完全相等，未改算式/PRNG顺序；这与浮点轨迹逐位一致是不同证据。

三次诊断命令墙钟分别99.029、70.021、90.706秒，合计259.756秒。第一次完成采集但数值/归因验收失败；第二次暴露HLO默认导出省略metadata的问题；第三次在新标记版编译阶段由命令预算停止。之后仅做CPU恢复和回归，未追加GPU采集。保留原失败结果，没有调宽阈值。

## 值得尝试的通用优化

1. **先量实际求解迭代，再做求解器预算对照。** 记录Newton/线搜索实际迭代、约束残差、失败/终止和回报，再比较通用iterations、ls_iterations、tolerance组合。当前已使用Newton；不能仅因100/50上限较高就直接降低。MJX官方将求解器迭代和接触工作量列为性能调节方向，但稳定性要由任务验证。[MuJoCo MJX性能说明](https://mujoco.readthedocs.io/en/stable/mjx.html#performance-tuning)
2. **比较可复用的物理后端/循环执行方式。** 热点是许多短物理kernel，值得做相同初态、动作、接触和约束的后端A/B，观察完整rollout吞吐及数值误差。MJX-Warp是可研究的候选，尚未接入或证明收益；其后端能力/模型支持须先核对。现有trace已经出现CUDA command buffer，不能把“打开command buffer”当成尚未使用的优化。[MJX后端说明](https://mujoco.readthedocs.io/en/stable/mjx.html)、[JAX GPU性能说明](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
3. **并行规模与训练batch一起设计。** 当前128环境可能值得进行吞吐对照，但Brax要求batch_size×num_minibatches能被num_envs整除。直接改256环境与当前128×1不兼容；改变batch还会影响更新频率和学习行为，须同时核对样本效率。

GAE、critic大小、反向传播及优化器尚无可靠的分别占比。当前critic为512×4、actor为64×2只是结构事实，不能据此断言critic是瓶颈。PPO算子已由JAX/XLA编译；目前没有证据支持再用C++重写整套PPO。已有C++评估在8环境的收益也不能直接外推到128环境的随机训练采样。若考虑扩展原生采样，应先验收通用轨迹/随机策略合同与完整批量吞吐。

## 数据驻留与搬运复查

生产Brax在同一个`pmap(training_epoch)`中以`lax.scan`执行环境采样、轨迹整理和PPO更新；环境状态、当前轨迹及优化器状态由设备循环携带。CPU参数快照用于停止保存/检查，并未作为每物理步的CPU往返数据源。增加并行环境和当前轨迹批可以利用更多显存；单纯增加显存预分配不等于减少拷贝或提高吞吐。

生产日志对搬运相关范围已有部分计时：三次`evaluation.params_to_host`合计14.04毫秒，包含参数转移和布局转换；`callback.snapshot_and_checks`合计1.440秒（进程129.526秒的1.11%），包含设备已就绪后的CPU快照及校验，尚未单列纯拷贝。`evaluation.reset_to_host`的4.998秒含首次reset/JIT，不能全算作拷贝。等待设备完成和取回数据要分别计量；`block_until_ready`本身不要求把结果搬回CPU。[JAX异步调度说明](https://docs.jax.dev/en/latest/async_dispatch.html)

仅用CPU重读同一次已保存trace，限定三个主机epoch窗口，并只取带`memcpy_details`且不带`kernel_details`的明确记录，得到：

| 设备实际拷贝 | 次数 | 总字节 | 设备区间并集 |
| --- | ---: | ---: | ---: |
| GPU→CPU（D2H） | 2,175 | 2,175 | 2.675毫秒 |
| GPU内部（D2D） | 6,940 | 42,639,360 | 6.997毫秒 |

这些窗口没有记录到明确的CPU→GPU（H2D）拷贝，不能外推为整个训练生命周期零H2D。D2H每次只有1字节，提示应追查循环控制/同步信号，其具体来源尚未关联确认；没有看到这些窗口在大量转出完整训练数据。D2H的主机API区间合计49.18毫秒，可能包含提交/等待，不能与设备时间直接相加或当作纯传输时间。小数据量也不能排除频繁同步造成的延迟。`memcpy32_post`等带`kernel_details`的设备kernel未算作PCIe传输；stream名称同时包含Compute和Memcpy，不能按stream名字将整条stream都判为拷贝。

原始复查脚本/结果在本机`tasks/results/ppo-stage-profiling/inspect-transfers.py`和`transfer-events.json`。采集扰动和诊断适配边界沿用上文，尚无全生产流程的H2D/D2H字节数、纯拷贝与同步等待三项独立汇总。下一次并行规模对照宜同时补齐这些字段；可用[JAX transfer guard](https://docs.jax.dev/en/latest/transfer_guard.html)查显式及隐式转移，但它记录调用来源，不能替代设备拷贝计时。

## 复用、验证和接手

入口见[README](README.md)，标记/异步/区间并集合同见[MEASUREMENT](MEASUREMENT.md)。CPU离线重放示例：

```bash
"$TENNY_PYTHON" -B trace_summary.py "$PROFILE_TRACE" \
  --window-name ppo.host_epoch --compiled-hlo "$REFERENCE_HLO" \
  --annotated-mlir "$ANNOTATED_MLIR" --output "$SUMMARY_JSON"
```

最终CPU回归56项：55项通过，1项可选本机源码快照测试跳过；覆盖AST保持/恢复、GPU事件识别、时间单位、重叠/unknown/mixed、HLO调用图与受限符号映射，以及忽略SIGINT/SIGTERM后的进程强制退出。CLI帮助和shell语法检查通过。最终可复用启动器的CPU取消行为已测，修正后的完整GPU采集尚未成功验收。

本机证据目录 `tasks/results/ppo-stage-profiling/` 不入库：`trace-summary-hlo.json`是最终CPU恢复结果，`attempt-1/result.json`保留首次失败，`launch.json`是第三次启动回执；三者不能混为同一次成功运行。`verification.json`记录生产/模型/导出/Brax身份未变，控制台active=false、can_resume=true、预览关闭。WSL保留原始trace、compiled HLO和lowered MLIR。接手时先重放已有证据，不为调解析器反复占用GPU。
