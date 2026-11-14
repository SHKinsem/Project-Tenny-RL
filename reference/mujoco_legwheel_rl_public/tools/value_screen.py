#!/usr/bin/env python
"""value 网络搜索 stage-1 初筛:V1–V4 × seeds {0,1} @ 1e8。

依据 docs/network_structure_plan.md 第 3/4 节:
- policy 锁 [64,64](netstruct 结论),只改 value_hidden_layer_sizes;
- σ 固定 0.5,其余全部锁 baseline;
- V0 [512]x4 是 baseline,1e8 初筛数据直接复用 tuning/netstruct/trials/A0-s{0,1}
  (policy [64,64] + value 默认 + 全 baseline,配置完全一致),不重跑;
- 每 trial 末尾 worker 内 deploy 口径加测,排名用无权重和;
- 记录 wall_time_s 观察训练吞吐变化(value 变大的代价)。

产物:tuning/netstruct/trials/V{1..4}-s{0,1}/。
用法:.venv-local/bin/python tools/value_screen.py
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec_mod = importlib.util.spec_from_file_location(
    "tune", os.path.join(ROOT, "scripts", "tune.py"))
tune = importlib.util.module_from_spec(_spec_mod)
_spec_mod.loader.exec_module(tune)

STUDY_DIR = os.path.join(ROOT, "tuning", "netstruct")

# 编号/结构与 plan 文档第 3 节一致;V0=baseline 复用 A0 数据
CANDIDATES = {
    "V1": [1024, 1024],                  # 加宽减深
    "V2": [1024, 1024, 1024, 1024],      # 容量上限
    "V3": [512, 512, 512, 512, 512, 512],  # 加深
    "V4": [256, 256],                    # 反向验证 value 是否过剩
}


def main():
    os.makedirs(STUDY_DIR, exist_ok=True)
    jobs = []
    for cid, layers in CANDIDATES.items():
        for s in (0, 1):
            jobs.append({
                "trial_id": f"{cid}-s{s}",
                "stage": "value-screen",
                "spec": dict(
                    tune._spec(s, 10, 1e8, {},
                               {"value_hidden_layer_sizes": layers}),
                    measure_deploy_action_rate=True,
                ),
            })
    print(f"[value-screen] {len(jobs)} trials @1e8 x 10 evals,gpus=0,1 "
          "(V0 复用 netstruct A0)", flush=True)
    tune.run_scheduler(STUDY_DIR, (0, 1), tune.static_jobs(jobs), 10.0)


if __name__ == "__main__":
    main()
