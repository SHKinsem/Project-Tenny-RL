# 配置与逻辑对比：`target_direct` @ 33238d9 vs 当前工作区

对比对象：

- 旧快照：git 提交 `33238d9`（分支 `target_direct`，"add save_cpu_checkpoint"），配置与逻辑位于根目录 `legwheel_env.py`、`ppo_test.py`、`domain_randomize.py`、`locomotion_params.py`。
- 当前快照：当前工作区代码，已重构为 `legwheel/` 包（`legwheel/config.py`、`legwheel/env.py`、`legwheel/rewards.py`、`legwheel/randomization.py`、`legwheel/training/` 等）。

> **2026-08-25 修订**：依赖栈已升级为 brax v0.14.2（git tag）+ jax 0.9.2 +
> mujoco/mjx 3.12.0 + playground 0.2.0（锁定 jax 0.9.2 的原因：brax v0.14.2
> 使用了 jax ≥ 0.10 中已移除的 `jax.device_put_replicated`）。本文已按升级后
> 代码重新核对，受影响结论集中在第三节与总结；另注意 mujoco 3.12 与旧快照
> 使用的 3.2.7 在物理求解器数值上可能有细微差异，同种子跨版本逐 bit 复现不适用。

---

## 一、配置参数差异（旧 `legwheel_env.py: default_config()` → 新 `legwheel/config.py`）

### 新增参数

- `episode_time=6.0` — `episode_length` 改为由 `episode_time / ctrl_dt` 计算（结果仍是 600，改为时长驱动）
- `cmd_start_delay_s=0.2` — 首次采样指令前的延迟
- `action_delay_steps=1` — 动作延迟步数（模拟控制链路延迟，10ms @100Hz）
- `reward_config.termination_penalty_enable=False`、`termination_penalty_value=-10.0` — 非正常终止惩罚开关及数值

### 删除参数

- `state_history_len=1`
- `use_accel=False`
- `action_range.wheel=[0, 0]`（现在 `action_range` 只保留 `thigh`）

### 实现方式变化（数值未变）

- `command_change_interval`：由 `episode_length // 2`（=300）改为 `3.0 / ctrl_dt`（=300），从"半个 episode"改为固定 3 秒
- `_XML_PATH`：由脚本目录相对路径改为从 `legwheel.assets.XML_PATH` 导入（assets 目录）

### 完全未变

- `ctrl_dt=0.010`、`sim_dt=0.0025`、`action_repeat=1`、`obs_history_len=1`、`wheel_radius=0.0195`
- `vel_pid` 全部子参数（kp=0.005、torque_limit=0.04、target_scale=40.0 等）
- 各类观测/行为开关（sinpitch_zero、sinyaw、last_act_obs 等）
- `action_scale` / `action_bias`、`noise_config`、`push_config`、`cmd_config`、`obs_normalize`
- `reward_config` 中除终止惩罚外的所有系数

---

## 二、环境/样本层面（`legwheel_env.py`@33238d9 → `legwheel/env.py`）

### 实质性行为差异

1. **指令采样 zero-margin 修复（影响最大）**
   - 旧：`sample_command` 用 `tmp > margin` 判定，导致**所有负向 lin_vel_x / ang_vel_yaw 指令一律被清零**（指令范围 [-0.2,0.2] / [-1.5,1.5]，约一半样本实际为零指令）。
   - 新：`legwheel/env.py:611,614` 改为 `abs(tmp) > margin`，负向指令保留，指令分布变为对称双向；下游 `wheels_sinposition_start`/`sinyaw_start` 基准更新和奖励门控现在会真正因负指令触发。

2. **新增动作延迟**：`action_delay_steps=1`，实际作用于仿真（thigh 目标、wheel 目标/PID）的是上一步动作（`env.py:253-260`）。注意 obs 里的 `last_act` 仍是策略原始输出而非实际执行动作；首步执行全零缓冲动作。

3. **首次指令重采样提前**：硬编码 `step == 100`（1 秒）→ `cmd_start_delay_s=0.2s` 换算的 step==20（`env.py:371-376`）。

4. **新增 episode_step 计数 + 终止惩罚机制**（`env.py:187, 392, 397-407, 417`）：`done & episode_step < episode_length` 时加 `termination_penalty_value(-10.0)`。默认 `enable=False`，默认行为与旧版一致。

5. **RNG 流改变**：噪声采样 key 拆分次数/顺序不同（旧 `_get_obs` 每处 `split(rng, 3)` 且 wheel_ctrl_vel_error 复用 key；新全部二路拆分并多拆一次，`env.py:487`），同种子逐样本不可复现（分布不变）。

### 逻辑等价 / 仅重构

- reset/step 主流程、推力扰动（方向 U(0,2π)、幅值 U(magnitude_range)、触发条件）、轮子 PID 速度环（kp=0.005、限幅 ±0.04、iters=ctrl_dt/sim_dt 次子步）、终止条件（isnan、sinpitch<-0.6 或 >0.85、|sinroll|>0.6）、观测构成与顺序、奖励各项公式与权重（`exp(-|err|/sigma)`、zero-margin 门控、energy_cost、action_rate、权重 height=2.0/vel=2.0/roll=4.0/zero_keep=10.0/energy=-1.0/action_rate=-0.5/pitch_vel=-0.5）均逐行一致。
- 死代码/缺陷清理：旧 `_get_reward` deconflict 分支调用不存在的方法（do_deconflict=False 死路径）；旧 `action_size` 属性错误引用 `self.env`；未使用的 `yaw_to_quaternion`、`*_history` 采样器等删除；XML 加载由 `from_xml_string` 改为 `from_xml_path`。
- **2026-08-25 升级附带**：`mjx_env.init(model, qpos=...)` → `mjx_env.make_data(model, qpos=...)`（`env.py:146`，playground 0.2.0 移除了 `init`，两者签名兼容、行为等价）。

---

## 三、训练算法层面（`ppo_test.py` + `locomotion_params.py` → `legwheel/training/`）

### 前提澄清

`locomotion_params.py` 在旧快照中是**死代码**（`ppo_test.py` 从未导入它），旧训练实际生效值来自 `get_ppo_config()` + `curriculum_learning()` 覆盖。

### 实质性行为差异

1. **discounting 大变（算法层面最大差异）**
   - 旧 `ppo_test.py`：`discounting = exp(log(0.5)/(t_base·(1/ctrl_dt)))`，t_base 由 log(0.99)/0.003 推出 → **γ ≈ 0.9671**（半衰期约 0.2s，且硬编码 0.003 与实际 sim_dt=0.0025 不符）。
   - 新 `legwheel/training/ppo.py:237-249`：改为按时间半衰期定义，`discounting = 0.5^(ctrl_dt/half_life_s)`（`ppo.py:130` 的基值 0.99 会被覆盖，运行时日志会打印实际生效值）。
   - **当前取值（2026-08-25，`ppo.py:240`）**：`half_life_s = 0.5` → **γ = 0.5^0.02 ≈ 0.9862**（等价 50Hz 下 γ≈0.972）。
   - 此前曾用 `half_life_s = episode_time/2 = 3.0s` → γ ≈ 0.9977（等价 50Hz 下 0.995），后认为偏长（见下评估），调至 0.5s 半衰期。
   - 有效回报视界从旧版 ~0.2s 拉长到 ~0.7s（~1/(1-γ)≈72 步），实质改变 credit assignment。

**对该改动的评估（2026-08-25 讨论记录）**：方向大概率有益，且旧值更像公式瑕疵的副产品而非调参结果。理由：
   - 这是自平衡任务，摔倒/推扰恢复的后果在数秒尺度体现（push 间隔 1~5s、恢复约 0.5~1s、episode 6s），γ≈0.967 的有效视界仅 ~1/(1-γ)≈30 步 ≈0.3s，价值函数几乎看不见即将到来的摔倒，无法给"牺牲即时跟踪奖励换稳定"的动作正确归因——低 γ 导致短视策略的典型场景。
   - γ 提高后提前死亡损失的未来回报流明显变大（旧 ~0.3s → 当前 ~0.7s 量级；若用 3s 半衰期则 ~4s），等效于内置了死亡惩罚，正好补上 `termination_penalty_enable=False` 的缺位，与加终止惩罚机制的意图同向。
   - 提高 γ 的主要代价（advantage 方差变大）在本配置下很小：GAE 兜底 + 每更新约 60 万样本（8192 envs × unroll 75），方差被充分平均。
   - 旧公式硬编码 0.003 与实际 sim_dt=0.0025 不符，且从 log(0.99) 绕两层推出，0.967 更像错误产物。
   - 取值折中：`half_life_s=3.0s`（γ≈0.9977，等价 50Hz 下 0.995）比常见 locomotion 默认值（0.99，半衰期约 0.7s）偏长，前期收敛可能变慢、value loss 增大；当前取 `half_life_s=0.5s`（γ≈0.9862），介于旧值 0.967 与 0.9977 之间，兼顾长视界信用分配与收敛速度。如实验显示不合适，可在 0.5~3.0s 半衰期之间再调。
   - 最终结论应以对照实验为准：同一 seed 各跑一次对比学习曲线。

2. **checkpoint 能力随 brax v0.14.2 升级恢复**（2026-08-25 修订）
   - 现使用 brax v0.14.2（git tag 安装），`ppo.train` 原生支持 `save_checkpoint_path` / `restore_params` / `restore_checkpoint_path`。
   - `train_ppo`（`ppo.py:166-191`）仍保留按签名过滤 kwargs 的兼容层，但在 0.14.2 上不再丢弃任何参数：训练过程中按 eval 间隔落盘 orbax checkpoint（`ckpts/<目录>/<全局步数>/`），训练结束另有 `params_cpu.pkl`。
   - checkpoint 保存/恢复路径统一转绝对路径（`ppo.py:118, 234`）——orbax 拒绝相对路径，相对路径曾使保存崩溃在 `.orbax-checkpoint-tmp` 中间态。
   - 课程迭代 ≥ 2 的热启动（params 非 None，走 `restore_params`）可用，不再抛 ValueError。
   - `--restore ckpts/<目录>/<步数>` 已实测可用；必须指向具体的步数子目录（指到父目录会报 "No structure could be identified"）。
   - （历史注：仓库曾锁定 brax==0.12.1，该版 `ppo.train` 没有上述参数，训练中不落盘 checkpoint、热启动直接报错；升级后与旧快照假定的"训练机新版 brax"行为对齐。）

3. `XLA_PYTHON_CLIENT_PREALLOCATE`：旧强制 `"False"`，新 `os.environ.setdefault`（`ppo.py:25`）——外部环境已设置时不再强制关闭预分配。

4. 日志：旧 `_setup_logging()` 无 `force=True` 且在 import 时执行，可能为 no-op；新 `legwheel/utils.py:46` `setup_logging(force=True)` 且只在 `scripts/train.py` 入口调用，保证落盘。

5. progress 回调容错与 `std_episode_length`（2026-08-25 修订）：旧直接索引 `metrics['eval/avg_episode_length']`（缺失即 KeyError）；现标题只显示 metrics 中实际存在的指标（`progress.py:39-48`，不再以 nan 兜底）。brax ≤ 0.12 的评估器缺少 `eval/std_episode_length`，由 `ppo.py:28-73` 的兼容补丁补算 `np.std(episode_steps)`；brax 0.14+ 原生输出该指标（`acting.py` 同时上报 mean/std），补丁检测到后自动跳过。

### 完全一致（仅重构）

PPO 超参数两边相同：seed=0、num_envs=8192、batch_size=512、num_minibatches=16、unroll_length=75、num_updates_per_batch=4、num_evals=10、num_timesteps=4e8、entropy_cost=1e-2、max_grad_norm=1.0、learning_rate=1e-4（基值 3e-4 被覆盖）、reward_scaling=0.1、normalize_observations=True、episode_length=600、policy 网络 [64,64]、value 网络 [512,512,512,512]、policy_obs_key="state" / value_obs_key="privileged_state"。eval 流程也一致。

---

## 四、域随机化层面（`domain_randomize.py` → `legwheel/randomization.py`）

### 实质性行为差异

1. **随机化开关全开**：旧 `use_mass_ipos=False`、`enable_thigh_jnt_random=False`、`enable_wheel_jnt_random=False`（实际只随机地面摩擦 + 初始位姿）；新版全部置 `True`（`randomization.py:48-52`），thigh/wheel 的 frictionloss/armature/stiffness/damping 随机化真正生效。
2. **质量/质心随机化生效且范围非零**：`torso_ipos_range` [0,0] → [-0.003, 0.003]；`torso_mass_range` [0,0] → [-0.01, 0.01]；`body_mass_ratio_range` [0.9,1.1] 不变。
3. **新增执行器随机化**（旧版完全没有，`randomization.py:83-89, 178-196`）：thigh 执行器 kp ×U(0.9,1.1)、kv ×U(0.9,1.1)；轮电机 gear ×U(0.8,1.2)。
4. RNG 消耗序列改变 → 同 PRNGKey 逐样本不可复现对齐（分布等价、逐样本不等价）。

### 未变

地面摩擦 U(0.8,1.2)、初始 xy U(-8,8)、yaw U(-π,π)、pitch U(-π/8,π/8)、roll U(-π/18,π/18)、thigh ×U(0.95,1.05)（damping ×U(0.9,1.1)）、wheel ×U(0.5,2.0) 的范围与分布完全不变。

---

## 五、评估工具层面

### `batch_test`（旧 `batch_test.py` → `legwheel/evaluation/`）

- 失败判定阈值、stats、失败记录保存/绘图逻辑等价。
- policy 重推理回放步数：`episode_length * 2` → `episode_length`（`replay.py:228`）。
- 精确 vmap 重现触发条件收紧：要求 `use_saved_actions and batch_context_file` 同时成立。
- obs 布局解析适配新观测（`last_act_obs=True` 时追加 last_act 4 维）。
- 渲染管线重写：全帧实时（render_every=1）、自建 `mujoco.Renderer`、新增速度/push 箭头、轮-地摩擦力与摩擦系数 OSD。

### `visualize`（旧 `visualizeTest.py` → `legwheel/evaluation/visualize.py`）

- **删掉了 `cmd_prob=0.0` 的覆盖**：旧版评估禁用随机指令；新版 cmd_prob 保持默认 0.9，6 秒评估中途指令可能被环境重采样覆盖测试用例命令 —— 注意这可能影响对比测试的公平性，需确认是否有意为之。
- reset 种子 PRNGKey(1)+关随机初始化 → PRNGKey(2211)+开随机初始化。
- 命令覆写方式修复：原地改 dict → `state.replace(info={...})`（JIT 下可靠）。
- 测试用例集缩减为单个 FailTest，ckpt 改走 CLI `--ckpt`。

---

## 总结

- **训练上最关键**：γ 从 0.967 → 0.9862（半衰期 0.2s → 0.5s，改为按时间半衰期定义）；域随机化从"仅摩擦"到"全开 + 新增执行器随机化"。
- **样本上最关键**：负向指令被清零的 bug 修复（指令分布对称化）；新增动作延迟 1 步。
- **需注意**：评估脚本不再禁用随机指令。
- **已解决（2026-08-25）**：训练栈升级 brax v0.14.2（git tag）+ jax 0.9.2 后，训练中 orbax checkpoint 正常落盘、`--restore` 与课程热启动可用、progress 曲线标题正常显示 `avg/std_episode_length`（旧 0.12.1 下 checkpoint 不落盘、std 显示 nan 的两个问题不复存在）。
