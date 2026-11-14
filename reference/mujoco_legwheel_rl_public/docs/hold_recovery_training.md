# hold_recovery 消亡史

日期:2026-09-01(移除提交)
相关:`docs/stand_tall_extend_issue.md`(原始病灶)、`docs/hold_position_heading.md`(双锚点方案本体)、`docs/hold_recovery_first_training_report.md`(首次训练实测记录)

> **代码已于本提交整体移除。配置键 `hold_recovery.*` 不再存在,旧 override
> (如 `--override hold_recovery.enable=true`)会报"未知配置路径"而不是被静默忽略。**

## 1. 诞生(2026-08-31,89cc4b8):为"随机站起伸直会跑"打的训练侧补丁

病灶:viewer 随机站起后执行"伸直"指令,机器人不收腿反而跑动、左右腿不齐
(`docs/stand_tall_extend_issue.md`)。当时的归因是**训练分布太窄**:

- PPO 只见到 `env.reset(rng)` 规范站姿,停车锚点误差从 ~0 起,策略只学会
  "误差很小时按住",没学过"误差很大时收回来";
- 奖励 `exp(-|e|/σ)` 在 |e|≳1.5 已饱和,大误差区没有梯度;
- 推理侧 `hold_error_clip`(只夹观测、不动锚点)只是安全网,不产生恢复能力。

于是按"训练分布加宽"的思路设计了 hold_recovery,三个组成部分
(**奖励坡 → 锚点注入 → 给够收回的时间**,总开关 `hold_recovery.enable`):

1. **奖励坡**:位置/朝向项在大误差区叠加线性代价
   (`-_hold_linear_cost`,e=0 时与原 exp 相同),外加 `height_split`(|L−R|)惩罚;
2. **锚点注入**:进入停车上升沿以概率 `inject_prob` 把位置或朝向锚点
   (一次只偏一路)拉开 U(±E),注入后 `inject_hold_steps`(150 步)强制零
   vx/yaw 指令(hold lock),让伺服有时间收;
3. **时间与初始化**:训练 reset 改随机姿态(`train_random_init`)、开局零指令、
   首个非零命令推迟到 1.5s(`cmd_start_delay_s`),专门覆盖"站起 + 冻原点"。

## 2. 转折(同日,a7fd419):双锚点重构证明它是补丁

同天的位置/朝向双锚点重构(`docs/hold_position_heading.md`)证明:真正的病根
不是"训练分布窄",而是**旧三锚点机制(轮相位锚)过约束**——轮相位同时被
位置、航向、roll 三个目标驱动,伸直即产生 ~7.5 rad 不收的相位误差。
双锚点(位置 Δs + 朝向 Δψ,刷新规则按指令通路分离)在机制上消除了这个病:
**不开 hold_recovery,仅靠新语义,随机站起 8/8 存活、伸直不再跑**
(见首次训练报告 §6.3 repro 修复版实测)。

hold_recovery 由此被重新定性:它治的"大误差不会收"在双锚点语义下已不是
主要矛盾,它是建立在错误归因上的补丁。

## 3. 死亡证据(首次训练与反事实实验)

实测与归因实验(`docs/hold_recovery_first_training_report.md` §3/§6,
`tools/analyze_hold_causality.py` E1-E3):

- **开 hold_recovery 训练,策略退化成"后仰坐地冻结"**:holdrec-step2(5e8)
  batch_test 成功率 87.9%、return ~3873,相对 baseline(94.82%/6265)全面
  失血;freeze 姿态定量 sinpitch ≈ −0.581,恰好卡在 termination 下限 −0.6
  内侧——"摔线之内最省力的存活姿态";
- **根因 = 训练分布失衡 + 姿态经济学**:随机初始化站起 + 1.5s 零指令 +
  注入锁定,把零指令门控激活占比推到 41%(正常协议 8.5%),"别动"的时间
  占比与收益被放大;`pitch_scale=0`(pitch 零权重)+ termination −0.6 宽松,
  使后仰坐地成为免费的生存吸引子;
- **反事实对照(E1-E3)**:E1 同轨迹新旧公式四方格打分等分(±0.5%),
  总差距 99% 是行为性;E2 机动污染探针证明"机动污染→惩罚机动"的机制链
  不存在;E3 静止收益核算证明新定义未放大"不动"的收益;
- **判定性消融(终审)**:同一套新 obs/锚点/reward 语义,仅关
  hold_recovery 从零重训(newobs-norecovery,4e8),freeze 消失、分项全面
  恢复并反超 baseline(**95.02% vs 94.82%、6262.9 vs 6265**);
- **注入通道本身也没起作用**:实测注入每 episode 仅触发 0.01~0.05 次
  ("进入停车上升沿"在 6s episode 内稀少),基本等于没注入。

结论:hold_recovery 既无必要(双锚点已治好原病)、又有大害(训练分布
把策略带进冻结吸引子)、自身还接近空转(注入率~0)。用户决定:移除。

## 4. 教训

1. **不要在机制性疾病上打分布补丁**。先证明机制没问题,再谈加宽分布;
   归因错了,补丁越厚偏得越远。
2. **奖励经济学会找到最省力的存活姿态**。任何"免费"的姿态(pitch 零权重 +
   宽松 termination)都会被分布失衡放大成吸引子;诊断项(权重 0 的分项)
   必须进报告监控,它往往是退化的第一信号。
3. **稀疏注入等于没注入**。触发率 0.01~0.05 次/episode 的训练通道,
   在短训预算内信噪比为零,设计时就要先算触发频率。
4. **反事实对照是归因利器**。同一份轨迹用两套公式打分(E1 四方格),
   能干净地区分"公式错了"还是"行为变了";判定性消融(只翻一个开关重训)
   是终审。

## 5. 移除清单(本提交)

- `legwheel/config.py`:`hold_recovery` 配置块整体删除;
- `legwheel/env.py`:锚点注入(`_maybe_inject_hold_error`)、hold lock
  (`_apply_hold_lock`/`hold_lock_remain`)、`train_random_init` 耦合
  (reset 恢复默认规范站姿,显式 `random=True` 的 viewer/batch_test 路径不变)、
  `cmd_start_delay_s` 耦合(恢复只用全局 0.2s)、`track/hold_inject` 指标;
- `legwheel/rewards.py`:`_hold_linear_cost` 及两处调用、`height_split`
  惩罚与 `_cost_height_split`/`reward/height_split` 指标、`_hold_recovery_enabled`;
- `tests/test_env.py`:`test_hold_recovery_*` 三个测试删除;
- `tools/analyze_holdrec_reward.py`:trainproto 口径下线(clean/standard 对新
  ckpt 保持可用);
- **保留**:位置/朝向双锚点(position_start/sinyaw_start)、Δs/Δψ 观测与门控、
  `hold_error_clip`(推理侧安全网,与 hold_recovery 无关)、训练误开
  `hold_error_clip` 的硬防护——这些是双锚点方案本体。
