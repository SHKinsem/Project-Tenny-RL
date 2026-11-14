"""通用工具:计时、日志、checkpoint 保存、GPU/EGL 环境配置。"""

import logging
import os
import pickle
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import jax


@contextmanager
def elapsed_timer(label: str = "Running"):
    """Context manager that prints elapsed time at 1 Hz in a background thread."""
    stop_event = threading.Event()
    start_time = time.time()

    def _reporter():
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"\r[{label}] Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}",
                end="",
                flush=True,
            )
            stop_event.wait(1.0)

    thread = threading.Thread(target=_reporter, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        elapsed = time.time() - start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n[{label}] Total elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")


def setup_logging() -> None:
    """配置 logging,写入 logs/training_<timestamp>.log。

    force=True:jax/brax 导入时可能已给 root logger 注册 handler,
    不加 force 的话 basicConfig 是 no-op,brax 的迭代日志会全部丢失。
    """
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=f'logs/training_{datetime.now().strftime("%Y%m%d-%H%M")}.log',
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def try_configure_mujoco_gpu_egl() -> None:
    """Best-effort MuJoCo EGL configuration.

    为 GPU(Colab/Linux 训练机)配置 EGL 渲染;macOS 或无 NVIDIA 环境下自动跳过。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        logging.info("nvidia-smi not found; skipping GPU EGL setup")
        return

    if result.returncode:
        raise RuntimeError(
            "Cannot communicate with GPU. "
            "Make sure you are using a GPU runtime (e.g., Colab GPU)."
        )

    # Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
    NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    try:
        if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
            os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)
            with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
                f.write(
                    """{
	\"file_format_version\" : \"1.0.0\",
	\"ICD\" : {
		\"library_path\" : \"libEGL_nvidia.so.0\"
	}
}
"""
                )
    except PermissionError:
        # On some environments, this location isn't writable.
        logging.warning("No permission to write NVIDIA ICD config; continuing")

    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    print("Setting environment variable to use GPU rendering:")
    os.environ["MUJOCO_GL"] = "egl"

    try:
        print("Checking that the installation succeeded:")
        import mujoco

        mujoco.MjModel.from_xml_string("<mujoco/>")
    except Exception as e:
        raise RuntimeError(
            "Something went wrong during installation. "
            "If using Colab, make sure GPU acceleration is enabled."
        ) from e

    print("Installation successful.")

    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags


def log_gpu_memory(tag: str = "") -> None:
    """Log GPU memory info: nvidia-smi 概览 + 每个 JAX 可见设备的池/实际用量。

    - nvidia-smi 行:每张物理卡的总显存/已用/利用率(含其他进程与 CUDA context)。
    - jax 行:每个可见设备的 XLA 内存池(bytes_limit=预预留上限,pool=已预留)
      与实际用量(in_use=当前,peak_in_use=峰值)——"占了多少"看 pool/limit,
      "实际用了多少"看 in_use/peak_in_use。
    无 NVIDIA GPU 时静默跳过。
    """
    prefix = f"[gpu-mem:{tag}]" if tag else "[gpu-mem]"
    gib = lambda b: b / 1024**3

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                idx, name, total, used, util = [p.strip() for p in line.split(",")]
                msg = (
                    f"{prefix} nvidia-smi gpu{idx} {name}: "
                    f"used {used}/{total} MiB, util {util}%"
                )
                logging.info(msg)
                print(msg, flush=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    for d in jax.devices():
        if d.platform != "gpu":
            continue
        try:
            stats = d.memory_stats() or {}
        except Exception:  # 旧版 jaxlib 可能不支持 memory_stats
            continue
        msg = (
            f"{prefix} jax {d}: pool {gib(stats.get('pool_bytes', 0)):.1f}/"
            f"{gib(stats.get('bytes_limit', 0)):.1f} GiB (reserved/limit), "
            f"in_use {gib(stats.get('bytes_in_use', 0)):.2f} GiB, "
            f"peak_in_use {gib(stats.get('peak_bytes_in_use', 0)):.2f} GiB, "
            f"peak_pool {gib(stats.get('peak_pool_bytes', 0)):.1f} GiB"
        )
        logging.info(msg)
        print(msg, flush=True)


def save_cpu_checkpoint(params, save_dir: str, filename: str = "params_cpu.pkl") -> None:
    """Save a CPU (numpy) version of the checkpoint via pickle.

    Args:
        params: JAX param pytree returned by ppo.train.
        save_dir: Directory to save the file in.
        filename: Name of the pickle file.
    """
    params_cpu = jax.device_get(params)  # recursively convert to numpy
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(params_cpu, f)
    print(f"Saved CPU checkpoint to {path}")
