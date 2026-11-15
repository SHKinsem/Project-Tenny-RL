# 可组合训练实验规格

**暂停草案：用户要求先仅做to-do，等并行网页工作完成后统一修改。此文件不是继续实现授权。当前交接与未完成状态以[待办清单](experiment-toggles-todo.md)为准。**

目标：近期算法改动可独立对照，配置可追溯，网页真实控制运行。

仅 leg_force_bc 新增 experiment_config 字典；腿位置＋轮扭矩继续保留原模式作参照。新实验默认绝对轮扭矩，旧 v3 模型缺字段时明确映射为残差头及其既有训练参数。续训采用保存配置，改配置须新实验，禁止静默重新解释权重。

统一接口 force_teacher.options：normalize_options(value=None)->完整JSON字典（严格类型/未知键/范围/组合校验），legacy_options()->v3既有配置，options_from_metadata(meta)->完整配置，option_schema()->网页字段/默认/预设。各模块可接受规范化 experiment_config；元数据与缓存绑定完整配置。

字段：wheel_head(absolute/residual)，wheel_residual_scale(0.05)，teacher_source(reachable/reference)，initial_state_perturbation(true)，balanced_sampling(true)，dagger_rounds(2)，wheel_loss_weight(1)，bc_gate_enabled(true)，fixed_normalization(true)，standing_only(true)，observation_noise(false)，noise_level(1)，external_pushes(false)，dynamics_randomization(false)，deterministic_eval(true)，bc_episodes(8)，bc_updates(1024)，bc_batch_size(256)，bc_learning_rate(.001)，bc_exploration_std(.01)，preparation_budget_seconds(300)，ppo_learning_rate(.00003)，entropy_cost(.001)，seed(0)。reference教师示范不支持学生访问状态DAgger，明确校验组合。新增尚未实现的物理参数范围随机化不冒充已有能力。

关闭门槛仍记录实际回放失败，不将其写成通过。配置冻结与源版本检查、产物哈希等属于完整性保障，不提供关闭。所有具体选项要说明是新训练生效、训练恢复继承还是仅影响评估；网页区分待启动配置和当前模型配置。比较用预设不是优劣结论，短验证不能声称收敛。

并行责任：core负责options/contracts/networks/policy/env/ppo；teacher负责pretrain/teacher；lifecycle负责training_control/console_train；web负责hybrid_web/preview_sync/dashboard及对应测试。不得覆盖其他改动；root负责集成验证与记录。
