#!/usr/bin/env python
"""value 网络搜索 stage-2 确认:初筛前 2 名(V2/V3)× seeds {0,1,2} @ 4e8。

依据 docs/network_structure_results.md 第 3 节:
- policy 锁 [64,64],只改 value_hidden_layer_sizes;σ 固定 0.5,其余锁 baseline;
- V0 不重跑,复用 run1-baseline 4e8 × 3 seeds 同口径数据。

产物:tuning/netstruct/trials/{V2,V3}-cs{0,1,2}/。
用法:.venv-local/bin/python tools/value_confirm.py
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
    "V2": [1024, 1024, 1024, 1024],
    "V3": [512, 512, 512, 512, 512, 512],
}


def main():
    os.makedirs(STUDY_DIR, exist_ok=True)
    jobs = []
    for cid, layers in FINALISTS.items():
        for s in (0, 1, 2):
            jobs.append({
                "trial_id": f"{cid}-cs{s}",
                "stage": "value-confirm",
                "spec": dict(
                    tune._spec(s, 20, 4e8, {},
                               {"value_hidden_layer_sizes": layers}),
                    measure_deploy_action_rate=True,
                ),
            })
    print(f"[value-confirm] {len(jobs)} trials @4e8 x 20 evals,gpus=0,1",
          flush=True)
    tune.run_scheduler(STUDY_DIR, (0, 1), tune.static_jobs(jobs), 10.0)


if __name__ == "__main__":
    main()
