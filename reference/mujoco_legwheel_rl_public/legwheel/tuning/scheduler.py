"""双卡 GPU 调度(计划 6.5.4):每卡一个 worker 槽位,按卡 flock 互斥。

- 钉卡:每个 worker 以 CUDA_VISIBLE_DEVICES=<k> 独占第 k 张卡,
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.1(约 9.6 GiB,为实测峰值 ~4 GiB 的两倍以上);
- 不堆并发:训练时 GPU util 已 90%+,SM 是瓶颈,每卡同一时刻只跑 1 个 trial;
- 冲突检测(双向保护):调度前用 pgrep + /proc/<pid>/environ 解析正式训练
  (scripts/train.py)占用的卡,只调度空闲卡;train.sh 反向检查 gpu*.lock。
"""

import fcntl
import json
import logging
import os
import subprocess

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)

WORKER_ENV_DEFAULTS = {
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_FLAGS": "--xla_gpu_triton_gemm_any=True",
}


def training_busy_gpus(all_gpus) -> set:
    """正式训练(scripts/train.py)占用的卡集合;未设 CUDA_VISIBLE_DEVICES 视为占全部。"""
    busy = set()
    try:
        out = subprocess.run(
            ["pgrep", "-f", "scripts/train.py"],
            capture_output=True, text=True, check=False,
        ).stdout.split()
    except FileNotFoundError:
        return busy
    for pid in out:
        if int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/environ", "rb") as fp:
                env = fp.read().decode(errors="ignore").split("\0")
            cvd = next(
                (e.split("=", 1)[1] for e in env
                 if e.startswith("CUDA_VISIBLE_DEVICES=")),
                None,
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        if cvd is None or cvd.strip() == "":
            busy.update(all_gpus)  # 未钉卡 = brax 会用全部可见卡
        else:
            busy.update(int(x) for x in cvd.split(",") if x.strip().isdigit())
    return busy


class GPUSlots:
    """每卡一个槽位: flock(tuning/<study>/gpu<k>.lock) + 至多一个运行中的 worker。"""

    def __init__(self, study_dir: str, gpu_ids=(0, 1)):
        self.study_dir = study_dir
        self.slots = {}
        for g in gpu_ids:
            lock_path = os.path.join(study_dir, f"gpu{g}.lock")
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            self.slots[g] = {"fd": fd, "locked": False}

    @property
    def gpu_ids(self):
        return list(self.slots)

    def acquire(self, gpu) -> bool:
        """尝试占用第 gpu 张卡(正式训练占用或其他 tune 实例持锁时失败)。"""
        if gpu in training_busy_gpus(self.gpu_ids):
            return False
        try:
            fcntl.flock(self.slots[gpu]["fd"], fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self.slots[gpu]["locked"] = True
        return True

    def release(self, gpu) -> None:
        if self.slots[gpu]["locked"]:
            fcntl.flock(self.slots[gpu]["fd"], fcntl.LOCK_UN)
            self.slots[gpu]["locked"] = False

    def release_all(self) -> None:
        for g in self.slots:
            self.release(g)


def launch_worker(gpu: int, study_dir: str, trial_id: str, spec: dict,
                  jax_cache_dir: str, python_exe: str) -> subprocess.Popen:
    """拉起一个 trial worker 子进程(6.5.3 子进程隔离)。"""
    trial_dir = os.path.abspath(os.path.join(study_dir, "trials", str(trial_id)))
    os.makedirs(trial_dir, exist_ok=True)
    spec = dict(spec, trial_dir=trial_dir)
    input_path = os.path.join(trial_dir, "input.json")
    with open(input_path, "w") as fp:
        json.dump(spec, fp, indent=2)

    env = dict(os.environ)
    env.update(WORKER_ENV_DEFAULTS)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs(jax_cache_dir, exist_ok=True)
    env["JAX_COMPILATION_CACHE_DIR"] = os.path.abspath(jax_cache_dir)

    stdout_path = os.path.join(trial_dir, "worker_stdout.log")
    stdout = open(stdout_path, "w")  # noqa: SIM115 - 生命周期跟随子进程
    logging.info("[gpu%d] launch trial %s -> %s", gpu, trial_id, trial_dir)
    return subprocess.Popen(  # noqa: S603
        [python_exe, "-m", "legwheel.tuning.worker", "--input", input_path],
        env=env, stdout=stdout, stderr=subprocess.STDOUT, cwd=_PROJECT_ROOT,
    )
