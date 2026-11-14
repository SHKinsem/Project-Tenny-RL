# legwheel_rl

腿轮混合机器人(LegWheelRobot)的强化学习项目:基于 MuJoCo MJX 仿真 + Brax PPO,
训练两轮腿式机器人做平衡、速度/偏航跟踪、高度与 roll 姿态控制。

项目已重构为标准 Python 包结构,**与运行目录无关**(可在任意 cwd 下运行,适配远程训练机)。

## 目录结构

```
legwheel_rl/
├── legwheel/                  # 主 Python 包
│   ├── config.py              # default_config():环境与训练默认配置
│   ├── env.py                 # LegWheelRobot MJX 环境(reset/step/obs/动作映射/传感器)
│   ├── rewards.py             # 奖励函数 mixin(_reward_* / _cost_* / _diff_*)
│   ├── randomization.py       # 域随机化(质量/质心/关节/执行器/摩擦/初始位姿)
│   ├── assets.py              # 包内资产路径解析与模型加载(cwd 无关)
│   ├── utils.py               # 计时/日志/checkpoint 保存/EGL 配置
│   ├── viewer.py              # 交互式策略查看器(GLFW 键盘控制)
│   ├── training/
│   │   ├── ppo.py             # PPO 配置、训练循环、课程学习、训练后推理
│   │   └── progress.py        # 训练进度回调与曲线绘制
│   ├── evaluation/
│   │   ├── runner.py          # 批量评估核心(vmap 并发测试)
│   │   ├── failures.py        # 失败记录数据结构与保存/加载/分布图
│   │   ├── rendering.py       # 评估视频渲染(OSD 叠加/速度/推力箭头)
│   │   ├── replay.py          # 失败精确重现(bit-identical vmap 重放)
│   │   └── visualize.py       # 命令用例评估(跟踪曲线+演示视频)
│   └── deployment/
│       └── extract.py         # checkpoint → 实机 C++ 参数导出
├── assets/                    # 机器人 XML、mesh、地形贴图(由 assets.py 定位)
├── scripts/                   # CLI 入口
│   ├── train.py               # 训练
│   ├── batch_test.py          # 批量测试 & 失败分析/重现
│   ├── visualize.py           # 命令用例评估
│   ├── viewer.py              # 交互查看器
│   └── export_params.py       # 部署参数导出
├── tools/                     # 一次性分析/生成工具(不属于包)
│   ├── terrain_image.py       # 地形高度图生成(输出被 assets/ 的 XML 引用)
│   ├── compute_wheel_inertia.py / render_hip_pose.py / plot.py
├── deploy/                    # 部署导出物(params.cpp.txt)
├── tests/test_env.py          # 环境单元测试
├── pyproject.toml             # pip install -e . 支持
└── requirements.txt           # 锁定依赖版本
```

## 模型结构

非对称 Actor-Critic(teacher-student 式),网络由 Brax `make_ppo_networks` 构建,
swish 激活,观测经 running statistics 归一化:

- **Actor(策略)**:输入 `state` 观测(30 维,带噪声本体感觉),MLP `[64, 64]`,
  输出 8 维(4 均值 + 4 std),NormalTanhDistribution,动作 ∈ [-1,1]^4
- **Critic(价值)**:输入 `privileged_state` 观测(63 维,含无噪声真值),
  MLP `[512, 512, 512, 512]`,输出 V(s)
- **动作语义**:动作 0/1 → 大腿位置目标 `(a-1)×0.25` rad(裁剪到 [-0.5, 0],位置伺服 kp=120/kv=80);
  动作 2/3 → 轮目标角速度 `a×40` rad/s,由底层 PI 速度环(kp=0.005,扭矩限幅 ±0.04 N·m)
  在每个控制步的 4 个仿真子步内转成轮力矩。小腿关节由等式约束联动,不直接控制。

## 快速开始

```bash
pip install -r requirements.txt   # Python 3.12
pip install -e .                  # 可选:任意位置 import legwheel
# 国内网络推荐 uv + 阿里云镜像(实测最快最稳,详见 requirements.txt 文件头):
#   uv pip install --python .venv/bin/python -r requirements.txt \
#     --index-url https://mirrors.aliyun.com/pypi/simple/

# 训练(checkpoint 保存到 ./ckpts/<时间戳>/)
./train.sh                                   # 一键启动(默认 1 轮 / 4e8 步,后台运行,GPU 配置已内置)
./train.sh --iters 2 --maxstep 2e8 --fg      # 透传任意 train.py 参数,--fg 前台运行
python scripts/train.py                      # 等价的手动方式(需自行设置 GPU 环境变量,见下)
python scripts/train.py --restore ckpts/20260222-1708/000403660800   # 续训

# 单元测试
python tests/test_env.py

# 批量测试 & 失败挖掘(4096 并发环境)
python scripts/batch_test.py --ckpt_path ckpts/20260222-1708/000403660800 --num_envs 4096
# 精确重现某个失败(见 --help 了解全部选项)
python scripts/batch_test.py --ckpt_path ckpts/... --replay \
    --failure_file failures/failures_xxx.pkl --failure_idx 0

# 命令用例评估(跟踪曲线 + 演示视频,输出 test/<时间戳>/)
python scripts/visualize.py --ckpt ckpts/20260222-1708/000403660800

# 交互查看器(本机有显示时)
python scripts/viewer.py --ckpt ckpts/...

# 导出实机部署参数
python scripts/export_params.py --ckpt ckpts/20260111-2109/000801792000
```

> **brax 版本说明**(2026-08 修订):本仓库使用 **brax v0.14.2(最新 tag)+ jax 0.9.2**。
> brax v0.14.2 的代码用了 `jax.device_put_replicated`,该 API 在 jax ≥ 0.10 中已移除,
> 因此 jax 锁定 0.9.2;brax 建议从 git tag 安装:`pip install "brax @ git+https://github.com/google/brax@v0.14.2"`。
> checkpoint 保存(`--save-dir` 或默认 `ckpts/<时间戳>/<全局步数>`)与恢复
> (`--restore ckpts/<目录>/<步数>`,注意要指向步数子目录)均已验证可用;
> checkpoint 路径会被自动转为绝对路径(orbax 要求)。
> GPU 训练需另装 CUDA 版 jax 并建议开启以下环境变量(预分配显存 + Triton GEMM):
>
> ```bash
> XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 XLA_PYTHON_CLIENT_PREALLOCATE=true \
> XLA_FLAGS="--xla_gpu_triton_gemm_any=True" python scripts/train.py ...
> ```
>
> **显存与多卡说明**(2026-08-26 补):`PREALLOCATE=true + MEM_FRACTION=0.85`
> 是启动时**预预留**总显存的 85% 作 XLA 内存池(防碎片/抖动),nvidia-smi 看到的
> "占满"是预留而非真实消耗——24GB 卡上同样会预占 ~20GB,实际需求远小于此。
> 且 XLA 对**所有可见 GPU** 都做预分配,而 brax PPO 只用 device 0 计算:
> 双卡机器上单训练任务会把两张卡都占掉 85%。只想用一张卡时加
> `CUDA_VISIBLE_DEVICES=0`;要两卡各跑一个任务,则分别用
> `CUDA_VISIBLE_DEVICES=0/1` 启动两个进程并把 `MEM_FRACTION` 降到 ~0.4。

## 关键配置(`legwheel/config.py`)

- `sim_dt=0.0025` / `ctrl_dt=0.010`(100Hz 控制,每控制步 4 个仿真子步),episode 6s
- `action_delay_steps`:动作延迟步数(默认 1 = 10ms,模拟控制链路延迟;0 关闭)。
  观测中的 `last_act` 与 action_rate 惩罚记录的是策略原始输出,延迟只作用于仿真
- `vel_pid`:轮子速度环参数(目标速度映射、kp、扭矩限幅、低通平滑)
- `cmd_config`:指令范围(lin_vel_x ±0.2 m/s、ang_vel_yaw ±1.5 rad/s、height、roll),
  死区判断用 `abs()`,正负向指令均有效
- `push_config`:随机推力扰动(1~5s 间隔,0~0.5 m/s)
- `noise_config`:观测噪声(joint_pos/joint_vel/linvel/gyro/pitch/roll)

## 域随机化(`legwheel/randomization.py`,训练时由 brax 按 env 批量采样)

- 全部连杆质量 ×U(0.9, 1.1),躯干附加质量 +U(-0.01, 0.01) kg,躯干质心 ±3mm
- 大腿关节 frictionloss/armature/stiffness ×U(0.95, 1.05)、damping ×U(0.9, 1.1);
  轮关节对应 ×U(0.5, 2.0)
- 大腿 position 执行器 kp/kv ×U(0.9, 1.1);轮电机出力 ×U(0.8, 1.2)
- 地面摩擦 ×U(0.8, 1.2);初始位置 ±8m、姿态随机

## sim2real 注意事项

策略只依赖 `state`(30 维带噪声观测)可部署,但其中 `pid_hist`/`pid_error`
两个分量依赖 `wheels_commands_history` / `last_target_wheel_vel` 状态,
实机底层速度环必须与 `vel_pid` 配置(映射、kp、限幅、平滑系数)逐位一致,
否则观测分布与训练不符。部署权重用 `scripts/export_params.py` 导出。

## 历史遗留

旧结构(根目录散脚本 + notebook)已在 `improvements/bugfix-cleanup` 分支清理,
所有删除的文件(notebook、go1/、cartpole 等)可从 git 历史找回。
