"""Bounded software-render probe; never imports or controls a training runner.

Run using the existing WSL virtualenv, with --runtime-root pointing at a frozen
release. It only reads /state and XML/assets. Results go to the explicit output.
There is no policy simulation here. Render timings isolate drawing and JPEG;
status trials reuse the deployed read logic without acquiring its manager lock;
scheduler trials use a synthetic clock and never construct a real simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.request


def source_digest(root):
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".xml"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
            count += 1
    return {"sha256": digest.hexdigest(), "files": count}


def state(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=5) as response:
        result = json.load(response)
    training = result["training"]
    return {"training": {k: training.get(k) for k in
            ("active", "phase", "run_id", "steps", "pid")},
            "preview": {k: result.get(k) for k in
            ("frames", "clients", "suspended", "paused", "generation")}}


def summarize(values):
    return {"median_ms": statistics.median(values) * 1000,
            "min_ms": min(values) * 1000, "max_ms": max(values) * 1000}


def probe(args):
    # Set these before importing any OpenGL, NumPy or MuJoCo module.
    os.nice(19)
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {available[-1]})
    os.environ.update(MUJOCO_GL="egl", LIBGL_ALWAYS_SOFTWARE="1",
                      GALLIUM_DRIVER="llvmpipe", LP_NUM_THREADS="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    # The experiment does not need access to CUDA devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    before = state(args.port)
    if before["training"]["active"]:
        raise RuntimeError("Render probe skipped because training is active")
    source_before = source_digest(args.runtime_root)
    result = {"schema": 1, "scope": "static pose, software rasterization only",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "before": before, "source_before": source_before,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0)}
    import mujoco
    from PIL import Image
    from OpenGL.GL import glGetString, GL_RENDERER

    model = mujoco.MjModel.from_xml_path(str(args.runtime_root / "assets" /
                              "legwheel_robot2_stl_nolinkage6_terrain.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("root").id
    camera.azimuth, camera.elevation, camera.distance = 35, -18, .65
    cases = [(800, 500, 80, True), (800, 500, 80, False),
             (640, 400, 80, True), (640, 400, 80, False),
             (400, 250, 80, False), (800, 500, 65, True)]
    result.update(mujoco_version=mujoco.__version__, cases=[])
    started = time.monotonic()
    for width, height, quality, effects in cases:
        if time.monotonic() - started > 20:
            result["budget_exhausted"] = True
            break
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            # EGL is only a context provider. Verify software rasterization.
            renderer._gl_context.make_current()
            gl_name = glGetString(GL_RENDERER).decode()
            if "llvmpipe" not in gl_name.lower():
                raise RuntimeError("Refusing a renderer that is not llvmpipe")
            result["gl_renderer"] = gl_name
            stages = {key: [] for key in ["forward", "scene", "render_readback", "jpeg", "total"]}
            sizes = []
            for sample in range(6):
                t0 = time.perf_counter()
                mujoco.mj_forward(model, data)
                t1 = time.perf_counter()
                renderer.update_scene(data, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(effects)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(effects)
                t2 = time.perf_counter()
                pixels = renderer.render()
                t3 = time.perf_counter()
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format="JPEG", quality=quality)
                t4 = time.perf_counter()
                if sample:
                    for key, duration in zip(stages, [t1-t0, t2-t1, t3-t2, t4-t3, t4-t0]):
                        stages[key].append(duration)
                    sizes.append(len(buffer.getvalue()))
            result["cases"].append({"width": width, "height": height, "quality": quality,
                "shadows_reflections": effects, "measured_frames": 5,
                "stages": {key: summarize(value) for key, value in stages.items()},
                "median_jpeg_bytes": statistics.median(sizes)})
    result["render_wall_seconds"] = time.monotonic() - started
    result["after"] = state(args.port)
    result["source_after"] = source_digest(args.runtime_root)
    result["source_unchanged"] = result["source_before"] == result["source_after"]
    return result


def probe_status(args):
    import cProfile
    import pstats
    os.nice(19)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(args.runtime_root))
    from training_control import TrainingController, _read
    from force_teacher import contract
    before = state(args.port)
    if before["training"]["active"]:
        raise RuntimeError("Status probe skipped because training is active")
    artifacts = args.artifacts
    record = _read(artifacts / "console/active.json")
    if not record.get("ended_at"):
        raise RuntimeError("Status probe requires a terminal record")
    source_before = source_digest(args.runtime_root)

    class ReadOnlyController(TrainingController):
        def _current(self):
            # No manager lock, liveness repair, training API or production write.
            if _read(self.root / "active.json") != record:
                raise RuntimeError("Training record changed; probe aborted")
            return record, _read(self._run_path(record) / "status.json"), False

    controller = ReadOnlyController(artifacts, args.original_root, artifacts.parent)
    original_metadata = contract.metadata
    records = []
    for label in ["baseline", "batch_metadata", "baseline", "batch_metadata"]:
        calls = []
        cache = {}

        def metadata(root=None):
            key = str(root)
            if label == "baseline" or key not in cache:
                calls.append(key)
                cache[key] = original_metadata(root)
            return cache[key]

        contract.metadata = metadata
        try:
            started = time.perf_counter()
            value = controller._status("leg_force_bc")
            seconds = time.perf_counter() - started
        finally:
            contract.metadata = original_metadata
        payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        records.append({"case": label, "seconds": seconds,
                        "metadata_computations": len(calls),
                        "result_sha256": hashlib.sha256(payload).hexdigest()})
    profiler = cProfile.Profile()
    profiler.runcall(controller._status, "leg_force_bc")
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats("cumtime").print_stats(16)
    return {"schema": 1, "scope": "isolated terminal status read and per-call metadata reuse",
            "trials": records, "identical_results": len({r["result_sha256"] for r in records}) == 1,
            "profile": buffer.getvalue(), "before": before, "after": state(args.port),
            "source_unchanged": source_before == source_digest(args.runtime_root),
            "note": "Prototype only; no production monkeypatch or deployment. Profiling overhead is separate from trial wall times."}


def probe_scheduler(args):
    from unittest.mock import patch
    sys.path.insert(0, str(args.runtime_root))
    import workbench_preview as module
    results = []
    for count, connected in [(1, 1), (4, 4), (4, 1)]:
        clock = [0.]

        class FakePreview:
            def __init__(self, clients):
                self.clients, self.frames = clients, 0

            def snapshot(self):
                return {"suspended": False, "clients": self.clients}

            def tick(self):
                self.frames += int(self.clients > 0)

            def release(self):
                pass

        previews = [FakePreview(int(i < connected)) for i in range(count)]
        group = module.PreviewGroup("leg_force_bc", object(), lambda _: previews[0])
        group.previews = {str(i+1): p for i, p in enumerate(previews)}

        class FakeWake:
            def wait(self, timeout):
                clock[0] += timeout
                if clock[0] >= 1 - 1e-9:
                    group.closed = True

            def clear(self):
                pass

        group.wake = FakeWake()
        with patch.object(module.time, "monotonic", lambda: clock[0]):
            group.run()
        results.append({"slots": count, "connected": connected, "synthetic_wall_seconds": clock[0],
                        "frames_per_slot": [p.frames for p in previews],
                        "simulated_seconds_per_slot": [.05*p.frames for p in previews]})
    return {"schema": 1, "scope": "deployed scheduler with zero-cost fake previews and synthetic time",
            "results": results, "note": "Not a measured hardware frame rate; .05 simulation seconds per frame comes from deployed Simulator.frame."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["render", "status", "scheduler"], default="render")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--original-root", type=Path)
    args = parser.parse_args()
    if args.kind == "status" and (args.artifacts is None or args.original_root is None):
        parser.error("status requires --artifacts and --original-root")
    result = {"render": probe, "status": probe_status, "scheduler": probe_scheduler}[args.kind](args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
