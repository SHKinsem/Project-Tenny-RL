# 正式预览性能：首轮测量与优化顺序

2026-11-15，渲染采样时间为20:45 HKT。研究对象为实际运行的工作台发布版 `workbench-20261115T114450Z`，并非当前 Windows 工作区里的旧入口或独立界面原型。

有明确优化空间。优先减少网页状态读取中的重复几何验证、修正多窗播放时钟，再降低画面成本。这些可以在独立预览/网页发布版中验证，无需修改训练算法、已有模型或训练源文件。本轮没有部署优化。

## 观察到的运行状态与边界

- 正式工作台的WSL服务为8771，旧控制台为8770；本机8095转到工作台。初查及实验前后，均报告 `active=false`、`phase=completed`，训练步数11,862,016。未发现仍在运行的训练runner。
- 工作台预览始终关闭，`frames=2100`、`clients=0`、`suspended=true`、`paused=true`、`generation=2`，测试没有打开其画面或申请训练策略快照。
- 未发送任何POST/训练控制请求，未停止或重启服务，未更改训练、预览生产源码或模型。实际发布版的125个Python/XML文件在渲染实验前后摘要相同：`4111a9ed0eb83c93c7dfcc46d38980769a6a5fe8f9549279d685db816d114b3f`。
- 本机短测限制在单个CPU逻辑核、nice19；渲染明确使用llvmpipe软件路径。没有训练GPU profiler、CUDA仿真或新训练。由于当时没有活跃训练，不能据此声称已经验证“训练吞吐完全不下降”。CPU/内存竞争仍需在后续实际训练中被动观察。

## 1. 网页状态读取在反复计算相同几何合同

直接在WSL回环读取正式服务，两次 `/state` 分别约1791、2214毫秒，响应53,317字节；预热后的 `/api/workbench/v1/viewports` 约2.04毫秒，1,314字节。它证明状态查询存在独立的慢路径，不能据此推断图像网络传输慢。

隔离进程调用发布版相同的状态读取逻辑，固定已结束的运行记录，不获取生产管理锁、不执行修复写入。调用链为：

`TrainingController._status → _latest → _runs → _validate_artifacts → force_teacher.contract.verified_report → metadata → scan_geometry`。

`leg_force_bc`的`hybrid=True`使`_latest`每次不使用已有缓存。当前11个候选目录中，9个进入完整教师合同验证；每个又重新计算相同源码对应的预期metadata和几何扫描。带profiler的独立读取中，`scan_geometry`累计2.323秒，占总2.708秒的约86%；此时间受profiler影响，只用于定位，不作为正常接口延迟。

对相同终态数据交替做两个baseline和两个原型试验：只在一次状态读取内部复用相同的预期metadata，所有逐模型元数据比较和checkpoint/export哈希验证继续执行。

| 方案 | 正常墙钟两次 | 每次metadata实际计算 |
| --- | --- | --- |
| 发布版原逻辑 | 1.006、0.985秒 | 9次 |
| 单次读取内部复用预期metadata | 0.1492、0.1494秒 | 1次 |

四次结果的规范化JSON SHA256完全相同。该CPU读取路径约快6.7倍、耗时降约85%。这是隔离读取逻辑的收益，并非已部署HTTP端点或整幅画面FPS提升。

实现方向：先将昂贵且同一源码下不变的预期合同计算从逐模型循环中提出来，再考虑有明确失效规则的模型目录缓存。每次批量读取要绑定同一源码代次；源/模型/保存回执变化时重新验证。训练开始、恢复和导入等真实操作继续走完整校验。应在网页适配层或独立发布版实现，避免修改进入旧模型源指纹的训练文件。不要直接永久缓存“模型有效”布尔值。

源码依据：发布版 `training_control.py:234`、`force_teacher/contract.py:36`、`hybrid/core.py:181`。前端 `workbench-ui/transport.mjs:25` 先等待 `/state`，因此这个慢路径还会推迟后续视窗状态请求；它们与渲染线程同处网页Python进程，存在争用解释器的可能，实际对FPS的影响尚未测量。

## 2. 多窗显示预算同时拖慢了仿真时间

发布版 `Simulator.frame` 每次推进约0.05秒仿真；`PreviewGroup.run`每轮依次tick所有视窗，再按 `.05 × 未suspend的视窗数` 等待。

用发布版真实调度器、零成本假视窗和合成时钟运行1秒，无需实际仿真：

| 保留视窗 | 有连接视窗 | 每窗帧数 | 每窗推进仿真秒数 |
| --- | --- | --- | --- |
| 1 | 1 | 20 | 1.00 |
| 4 | 4 | 各5 | 各0.25 |
| 4 | 1 | 5、0、0、0 | 0.25、0、0、0 |

这是调度行为的确定性复现，不是硬件实测FPS。即使计算足够快，四窗也会呈现约四分之一速；没有观众但仍未suspend的槽位也计入预算。暂停时也仍会反复渲染同一姿态，除非关闭或断开。

实现方向：策略/物理继续用原控制dt，独立确定仿真播放时间与显示帧率；不按显示帧数定义播放速度。预算按真正有连接、需要新帧的视窗分配。暂停且姿态/相机没有变化时复用JPEG，重置或相机变化时再绘制。只有在后台仿真确实跟不上时才报告实际播放倍率，不伪造实时速度。

这一修改需要CPU资源预算：只把一次frame中的物理步数乘以视窗数，会增加物理工作量，不能视作免费加速。同一策略/初态只看不同相机时，可以共享一份仿真；不同策略、随机初态或扰动实验仍需独立世界。

## 3. 单核软件绘图存在可测的显示侧收益

MuJoCo3.12.0，相同发布版XML/模型资源、固定姿态和相机。新建隔离Renderer，`LIBGL_ALWAYS_SOFTWARE=1`、`GALLIUM_DRIVER=llvmpipe`、`LP_NUM_THREADS=1`，实际GL字符串为 `llvmpipe (LLVM 20.1.2, 256 bits)`。每档1帧预热＋5帧测量，完整渲染矩阵约3.94秒。没有运行策略或MJX物理。

| 画面 | 阴影/反射 | JPEG质量 | 每帧总耗时中位数 | JPEG大小 |
| --- | --- | --- | --- | --- |
| 800×500 | 开 | 80 | 72.34毫秒 | 20,648字节 |
| 800×500 | 关 | 80 | 45.68毫秒 | 19,792字节 |
| 640×400 | 开 | 80 | 63.98毫秒 | 14,648字节 |
| 640×400 | 关 | 80 | 37.37毫秒 | 14,057字节 |
| 400×250 | 关 | 80 | 22.66毫秒 | 6,755字节 |
| 800×500 | 开 | 65 | 68.80毫秒 | 16,349字节 |

800×500基准中，绘图＋像素读回约71.02毫秒，JPEG约1.03毫秒，`mj_forward`约0.055毫秒。单独关闭阴影/反射，总耗时降约37%；400×250低效果画面相比基准降约69%。降低JPEG质量主要降低图像体积；不能把顺序测试的小幅总时间波动都归因于JPEG质量。

这些是限制单核的软件绘图数据，实际生产EGL可能使用不同设备/线程，不能直接当成当前工作台FPS。只读环境检查确认网页JAX为CPU，但 `MUJOCO_GL=egl` 本身不能证明绘图也不占GPU；本轮没有切换生产渲染后端。OSMesa库在当前系统未被发现，没有安装依赖；测试使用现有EGL的明确软件路径。

实现方向：保留大窗画质，给小窗按显示尺寸选择400×250或640×400，提供低效果档；暂停帧复用优先于继续编码同一图像。不为追求FPS直接把预览迁到训练GPU。

## 4. 下一阶段候选与尚未完成的验证

| 顺序 | 动作 | 与训练的关系 | 验收重点 |
| --- | --- | --- | --- |
| P0 | 单次模型列表读取只计算一次预期合同 | 先做独立网页适配层 | 原数据/错误结果一致；源码变化触发重验；实际HTTP延迟 |
| P0 | 暂停帧复用、剔除无观众槽位 | 仅预览调度 | 暂停时绘制次数归零；关闭/重开取消语义保持 |
| P1 | 小窗分辨率与阴影/反射档位 | 仅显示负载 | 相同动作/物理dt；画质可接受；1/4窗CPU与实际FPS |
| P1 | 仿真时间与显示帧率分开 | 仅预览，但须限制物理预算 | 4窗播放倍率正确；过载可见；不无限追赶积压 |
| P2 | 复用已有native MuJoCo评估能力做单世界预览 | 独立预览后端，保留训练MJX | 观测/动作头/历史/控制保持/reset/终止一致，漂移有界 |
| P3 | 服务端传姿态、浏览器绘图 | 改预览传输与渲染架构 | 几何/姿态映射、模型代次、延迟、接触标记和本机GPU争用 |

MJX-JAX官方明确指出单场景不是其优势，native候选有依据；但本轮没有测当前 `env.step`、首次JIT、策略推理或完整闭环，也没有实测native预览收益。已有C++评估接口是批量rollout返回结果，并非天然可暂停/续步的交互仿真；需要新适配或明确采用固定策略轨迹回放，不能直接反复调用rollout假装保持同一个世界。详见[官方资料与边界](preview-performance-primary-sources-2026-11-15.md)。

下一步可在独立网页发布目录验证P0/P1，保留现有训练runner与其源码位置。切换网页入口或部署尚未进行。本轮只新增研究记录和测量脚本，不修改现有脏工作区文件。

## 复现与证据

脚本：[preview_performance_probe.py](preview_performance_probe.py)。只在WSL使用现有虚拟环境；`status`和`render`在训练活跃时拒绝执行，`scheduler`只使用假视窗。

设置`TENNY_ROOT`为本仓库的WSL路径、`RELEASE_ROOT`为被测发布版的`src`路径后：

```bash
PY="$HOME/projects/tenny-reference/.venv-gpu/bin/python"
OUT="$TENNY_ROOT/tasks/results/preview-performance/20261115"
"$PY" -B "$TENNY_ROOT/experiments/torque200/research/preview_performance_probe.py" \
  --kind render --runtime-root "$RELEASE_ROOT" --port 8771 --output "$OUT/software-render.json"
"$PY" -B "$TENNY_ROOT/experiments/torque200/research/preview_performance_probe.py" \
  --kind status --runtime-root "$RELEASE_ROOT" --port 8771 \
  --artifacts "$HOME/projects/tenny-torque200/artifacts" \
  --original-root "$HOME/projects/tenny-reference" --output "$OUT/status.json"
"$PY" -B "$TENNY_ROOT/experiments/torque200/research/preview_performance_probe.py" \
  --kind scheduler --runtime-root "$RELEASE_ROOT" --output "$OUT/scheduler.json"
```

实际运行外层另设25/40秒子进程超时。脚本渲染矩阵设20秒软预算；已有训练随时可能被用户启动，不能把一次检查当作全程资源锁。原始测量位于忽略的`tasks/results/preview-performance/20261115/`：`software-render.json`、`status.json`、`scheduler.json`和初次`status-profile.txt`。不要为复现而新增训练或覆盖生产目录。
