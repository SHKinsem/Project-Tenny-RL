#!/usr/bin/env python
"""网络结构搜索 stage-2 确认:初筛前 2 名(P6/P7)× seeds {0,1,2} @ 4e8。

依据 docs/network_structure_plan.md 第 4 节与 network_structure_results.md 第 2 节:
- baseline A0 不重跑,直接复用 run1-baseline 4e8 × 3 seeds(配置完全一致);
- 只改 policy_hidden_layer_sizes,其余锁 baseline;σ 固定 0.5;
- worker 末尾 deploy 口径加测,口径与 run1-baseline 确认轮相同。

产物:tuning/netstruct/trials/{P6,P7}-c{s0,s1,s2}/。
用法:.venv-local/bin/python tools/netstruct_confirm.py
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec_mod = importlib.util.spec_from_file_location(
    "tune", os.path.join(ROOT, "scripts", "tune.py"))
tune = importlib.util.module_from_spec(_spec_mod)
_spec_mod.loader.exec_module(tune)

STUDY_DIR = os.path.join(ROOT, "tuning", "netstruct")

FINALISTS = {
    "P6": [48, 48, 48],
    "P7": [64, 32, 64],
}


def main():
    os.makedirs(STUDY_DIR, exist_ok=True)
    jobs = []
    for cid, layers in FINALISTS.items():
        for s in (0, 1, 2):
            jobs.append({
                "trial_id": f"{cid}-cs{s}",
                "stage": "netstruct-confirm",
                "spec": dict(
                    tune._spec(s, 20, 4e8, {},
                               {"policy_hidden_layer_sizes": layers}),
                    measure_deploy_action_rate=True,
                ),
            })
    print(f"[netstruct-confirm] {len(jobs)} trials @4e8 x 20 evals,gpus=0,1",
          flush=True)
    tune.run_scheduler(STUDY_DIR, (0, 1), tune.static_jobs(jobs), 10.0)


if __name__ == "__main__":
    main()
