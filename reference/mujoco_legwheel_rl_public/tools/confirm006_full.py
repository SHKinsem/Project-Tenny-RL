#!/usr/bin/env python
"""一次性:search2-006(被 ASHA 剪枝的 trial)配置补 4e8 × 3 seeds 全量确认。

confirm2 --candidates 只接受 study 中 COMPLETE 的 trial,006 是 pruned,
故直接复用 scripts/tune.py 内部件,口径与 confirm2 完全一致:
4e8、20 evals、measure_deploy_action_rate、constraints_ref=base2。
产物:tuning/run2/trials/cand006-s{0,1,2}/。
"""

import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec_mod = importlib.util.spec_from_file_location(
    "tune", os.path.join(ROOT, "scripts", "tune.py"))
tune = importlib.util.module_from_spec(_spec_mod)
_spec_mod.loader.exec_module(tune)

STUDY_DIR = os.path.join(ROOT, "tuning", "run2")
SRC = os.path.join(STUDY_DIR, "trials", "search2-006", "overrides.json")


def main():
    ov = json.load(open(SRC))
    env_ov, ppo_ov = ov["env"], ov["ppo"]
    base2 = tune._base2_ref(STUDY_DIR)
    stage0 = tune.read_json(os.path.join(STUDY_DIR, "stage0.json")) or {}
    constraints_ref = {"base2": base2,
                       "energy_budget": stage0.get("energy_budget")}
    jobs = [
        {"trial_id": f"cand006-s{s}", "stage": "confirm2",
         "spec": dict(tune._spec(s, 20, 4e8, dict(env_ov), dict(ppo_ov)),
                      measure_deploy_action_rate=True,
                      constraints_ref=constraints_ref)}
        for s in (0, 1, 2)
    ]
    print("[confirm006] cand006 × 3 seeds @4e8,gpus=0,1")
    tune.run_scheduler(STUDY_DIR, (0, 1), tune.static_jobs(jobs), 10.0)


if __name__ == "__main__":
    main()
