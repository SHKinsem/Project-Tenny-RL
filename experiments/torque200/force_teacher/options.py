"""Strict JSON experiment options; no simulator or ML imports."""
import math

DEFAULTS = dict(wheel_head="absolute", wheel_residual_scale=.05, teacher_source="reachable",
    initial_state_perturbation=True, balanced_sampling=True, dagger_rounds=2,
    wheel_loss_weight=1., bc_gate_enabled=True, fixed_normalization=True,
    standing_only=True, observation_noise=False, noise_level=1., external_pushes=False,
    dynamics_randomization=False, deterministic_eval=True, bc_episodes=8, bc_updates=1024,
    bc_batch_size=256, bc_learning_rate=.001, bc_exploration_std=.01,
    preparation_budget_seconds=300., ppo_learning_rate=.00003, entropy_cost=.001, seed=0)
CHOICES = {"wheel_head":("absolute","residual"), "teacher_source":("reachable","reference")}
BOUNDS = dict(wheel_residual_scale=(.000001,1.), dagger_rounds=(0,8), wheel_loss_weight=(.01,100.),
    noise_level=(0.,5.), bc_episodes=(4,64), bc_updates=(1,4096), bc_batch_size=(1,2048),
    bc_learning_rate=(1e-8,.01), bc_exploration_std=(.01,.5), preparation_budget_seconds=(.001,300.),
    ppo_learning_rate=(1e-8,.01), entropy_cost=(0.,1.), seed=(0,2147483647))
LABELS = dict(wheel_head="轮扭矩输出头", wheel_residual_scale="轮残差尺度", teacher_source="教师来源",
    initial_state_perturbation="教师采集初态扰动", balanced_sampling="回合与起步均衡采样",
    dagger_rounds="DAgger 轮数", wheel_loss_weight="轮动作损失权重", bc_gate_enabled="BC 站立门禁",
    fixed_normalization="固定观测归一化", standing_only="仅站立指令", observation_noise="观测噪声",
    noise_level="观测噪声倍率", external_pushes="外力扰动", dynamics_randomization="动力学随机化",
    deterministic_eval="确定性评估", bc_episodes="教师回合数", bc_updates="BC 更新数",
    bc_batch_size="BC 批量", bc_learning_rate="BC 学习率", bc_exploration_std="初始探索标准差",
    preparation_budget_seconds="准备预算（秒）", ppo_learning_rate="PPO 学习率", entropy_cost="熵系数", seed="随机种子")
HELP = dict(wheel_head="absolute直接预测绝对轮扭矩；residual在上周期实际轮扭矩的logit上预测修正，最终仍输出绝对扭矩。",
    wheel_residual_scale="只在residual头生效，缩放网络预测的轮扭矩logit修正；不改变电机限幅。",
    teacher_source="reachable在真motor学生状态上查询并投影老师；reference使用原位置伺服轨迹的可达示范，不能配合DAgger。",
    initial_state_perturbation="只控制教师采集时的轻微初始姿态/角速度扰动，不控制PPO动力学随机化。",
    balanced_sampling="按回合及起步/稳态阶段分配BC采样权重；关闭时按样本均匀抽样。",
    dagger_rounds="初始BC后最多执行的学生轨迹再标注轮数；所有轮共享准备总时间预算。",
    wheel_loss_weight="乘在BC轮动作平方误差上的权重，不改变腿动作权重或物理量上限。",
    bc_gate_enabled="要求三组不同初态各存活6秒才允许PPO；仅为短回放门槛，不认证稳定姿态或实机安全。关闭仍记录实际结果。",
    fixed_normalization="开启时冻结BC拟合的观测均值/尺度；关闭时由PPO持续更新Welford统计。",
    standing_only="仅控制训练环境命令采样：开启为零指令，关闭恢复既有移动指令采样。",
    observation_noise="在训练观测上添加既有传感噪声模型，幅度由噪声倍率控制。",
    noise_level="既有观测噪声的整体倍率；关闭观测噪声时不生效。",
    external_pushes="启用现有环境外力扰动；不改变电机动作或限幅。",
    dynamics_randomization="使用既有质量/重心/惯量、关节摩擦与阻尼、地面摩擦、电机强度和初始位姿随机化；不提供任意范围编辑。",
    deterministic_eval="开启时PPO评估使用动作分布的确定性输出；关闭时采样动作。",
    bc_episodes="初始教师采集回合数，每回合最多6秒。",bc_updates="每轮BC的优化更新次数。",
    bc_batch_size="每次BC更新抽取的样本数。",bc_learning_rate="BC Adam优化器学习率。",
    bc_exploration_std="BC完成后PPO动作分布的初始探索标准差。",
    preparation_budget_seconds="数据采集、BC及门禁共享的准备阶段总时间上限；不计入正式PPO训练时长。",
    ppo_learning_rate="正式PPO优化器学习率。",entropy_cost="PPO熵奖励系数，控制对探索的激励。",
    seed="实验采集、采样和训练使用的随机种子；恢复时继承保存值。")


def normalize_options(value=None):
    if value is None: value = {}
    if type(value) is not dict: raise ValueError("experiment_config must be a JSON object")
    unknown = set(value)-set(DEFAULTS)
    if unknown: raise ValueError("Unknown experiment options: " + ", ".join(sorted(unknown)))
    result = dict(DEFAULTS)
    for key, item in value.items():
        default = DEFAULTS[key]
        if key in CHOICES:
            valid = type(item) is str and item in CHOICES[key]
        elif type(default) is bool:
            valid = type(item) is bool
        elif type(default) is int:
            valid = type(item) is int
        else:
            valid = type(item) in (int,float) and math.isfinite(item)
        if not valid: raise ValueError("Invalid experiment option type/value: " + key)
        if key in BOUNDS and not BOUNDS[key][0] <= item <= BOUNDS[key][1]:
            raise ValueError("Experiment option outside supported range: " + key)
        result[key] = float(item) if type(default) is float else item
    if result["teacher_source"] == "reference" and result["dagger_rounds"]:
        raise ValueError("Reference teacher demonstrations do not support student-state DAgger; set dagger_rounds=0")
    return result


def legacy_options():
    """The explicit v3 training behavior, never the defaults for new experiments."""
    return normalize_options(dict(wheel_head="residual"))


def options_from_metadata(meta):
    if type(meta) is not dict or meta.get("mode") != "leg_force_bc":
        raise ValueError("Not a force experiment metadata object")
    schema = meta.get("teacher_schema")
    if schema == "leg-force-reachable-teacher-bc-v3":
        if "experiment_config" in meta or meta.get("policy_head") != "wheel_logit_residual_v1" or meta.get("wheel_residual_scale") != .05:
            raise ValueError("Conflicting legacy v3 semantics")
        return legacy_options()
    if schema != "leg-force-configurable-bc-v4": raise ValueError("Unsupported force experiment version")
    value = meta.get("experiment_config")
    if type(value) is not dict or set(value) != set(DEFAULTS):
        raise ValueError("Saved experiment must contain every option")
    return normalize_options(value)


def option_schema():
    fields=[]
    for key,default in DEFAULTS.items():
        kind = "select" if key in CHOICES else ("boolean" if type(default) is bool else "integer" if type(default) is int else "number")
        effect = "评估；恢复时继承保存值" if key == "deterministic_eval" else "新训练生效；恢复时继承保存值"
        group = ("policy" if key.startswith("wheel_") and key != "wheel_loss_weight" else
                 "environment" if key in ("standing_only","observation_noise","noise_level","external_pushes","dynamics_randomization") else
                 "ppo" if key in ("fixed_normalization","deterministic_eval","ppo_learning_rate","entropy_cost") else "preparation")
        field=dict(key=key,name=key,label=LABELS[key],type=kind,default=default,effect=effect,help=HELP[key],group=group)
        if key in CHOICES: field["choices"] = list(CHOICES[key])
        if key in BOUNDS: field.update(min=BOUNDS[key][0],max=BOUNDS[key][1])
        fields.append(field)
    return dict(version=1,defaults=normalize_options(),fields=fields,presets=[
        dict(id="absolute",label="绝对扭矩对照",values=normalize_options()),
        dict(id="residual",label="残差头对照",values=normalize_options(dict(wheel_head="residual"))),
        dict(id="reference",label="原版教师示范对照",values=normalize_options(dict(teacher_source="reference",dagger_rounds=0)))])
