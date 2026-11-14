#!/usr/bin/env python
"""网络结构搜索 stage-1 初筛:policy 候选 × seeds {0,1} @ 1e8。

依据 docs/network_structure_plan.md 第 2/4 节:
- 只改 policy_hidden_layer_sizes;value 固定 [512]*4,其余全部锁 baseline;
- σ/sigma_wheel 固定 0.5(不动);
- 每 trial 末尾由 worker 做 deploy 口径加测(deterministic + 域随机化 +
  push + 噪声 + 固定 σ=0.5),result.json 直接给出 attainment/action_rate,
  排名用无权重和 = Σ attainment - action_rate。

产物:tuning/netstruct/trials/<cid>-s{0,1}/。
用法:.venv-local/bin/python tools/netstruct_screen.py
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec_mod = importlib.util.spec_from_file_location(
    "tune", os.path.join(ROOT, "scripts", "tune.py"))
tune = importlib.util.module_from_spec(_spec_mod)
_spec_mod.loader.exec_module(tune)

STUDY_DIR = os.path.join(ROOT, "tuning", "netstruct")

# 编号/结构/MACs 与 plan 文档第 2 节一致
CANDIDATES = {
    "A0": [64, 64],        # baseline,6272 MACs
    "P1": [128],           # 4352
    "P2": [160],           # 5440
    "P3": [176],           # 5984(预算内单层最宽)
    "P4": [48, 48],        # 3936
    "P5": [56, 56],        # 5040
    "P6": [48, 48, 48],    # 6240(预算内最深)
    "P7": [64, 32, 64],    # 6272(瓶颈)
}


def main():
    os.makedirs(STUDY_DIR, exist_ok=True)  # GPUSlots 锁文件落在 study 根目录
    jobs = []
    for cid, layers in CANDIDATES.items():
        for s in (0, 1):
            jobs.append({
                "trial_id": f"{cid}-s{s}",
                "stage": "netstruct-screen",
                "spec": dict(
                    tune._spec(s, 10, 1e8, {},
                               {"policy_hidden_layer_sizes": layers}),
                    measure_deploy_action_rate=True,
                ),
            })
    print(f"[netstruct] {len(jobs)} trials @1e8 x 10 evals,gpus=0,1", flush=True)
    tune.run_scheduler(STUDY_DIR, (0, 1), tune.static_jobs(jobs), 10.0)


if __name__ == "__main__":
    main()
