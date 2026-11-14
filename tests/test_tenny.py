import sys, time, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenny import registry  # noqa: E402

PY = sys.executable
TASKS = {
    "echo": {"id": "echo", "name": "echo", "group": "t",
             "cmd": [PY, "-u", "-c", "import sys;print('hi',sys.argv[1])"],
             "args": ["{word}"],
             "params": [{"name": "word", "type": "str", "default": "x"}]},
    "sleep": {"id": "sleep", "name": "sleep", "group": "t",
              "cmd": [PY, "-u", "-c", "import time;print('go',flush=True);time.sleep(60)"]},
    "noisy": {"id": "noisy", "name": "noisy", "group": "t",
              "cmd": [PY, "-u", "-c", "[print(i) for i in range(6000)]"]},
    "num": {"id": "num", "name": "num", "group": "t",
            "cmd": [PY, "-c", "pass"], "args": ["{n}"],
            "params": [{"name": "n", "type": "int", "min": 1, "max": 5, "default": 2},
                       {"name": "mode", "type": "choice", "choices": ["a", "b"], "default": "a"},
                       {"name": "fast", "type": "flag", "flag": "--fast", "default": False}]},
}


from tenny import runner  # noqa: E402


def wait(job, timeout=15):
    end = time.time() + timeout
    while job.status == "running" and time.time() < end:
        time.sleep(0.05)


class RegistryTest(unittest.TestCase):
    def test_defaults_and_coercion(self):
        v = registry.resolve_params(TASKS["num"], {"n": "3", "fast": True})
        self.assertEqual(v, {"n": 3, "mode": "a", "fast": True})

    def test_bad_values_rejected(self):
        for bad in ({"n": "abc"}, {"n": 0}, {"n": 6}, {"mode": "z"}, {"evil": "1"}):
            with self.assertRaises(registry.TaskError, msg=bad):
                registry.resolve_params(TASKS["num"], bad)

    def test_build_args(self):
        v = registry.resolve_params(TASKS["num"], {"n": 4, "fast": True})
        self.assertEqual(registry.build_args(TASKS["num"], v), ["4", "--fast"])
        v = registry.resolve_params(TASKS["num"], {})
        self.assertEqual(registry.build_args(TASKS["num"], v), ["2"])

    def test_shell_metacharacters_stay_one_arg(self):
        v = registry.resolve_params(TASKS["echo"], {"word": "a; rm -rf /"})
        self.assertEqual(registry.build_args(TASKS["echo"], v), ["a; rm -rf /"])

    def test_real_toml_is_valid(self):
        tasks = registry.load()
        for t in tasks.values():
            self.assertTrue(t["name"] and t["group"], t["id"])
            if "script" in t:
                self.assertTrue((registry.ROOT / t["script"]).is_file(), t["script"])
            registry.resolve_params(t, {})  # 默认值必须自身合法


class RunnerTest(unittest.TestCase):
    def tearDown(self):
        runner.stop_all()

    def test_runs_and_captures_output(self):
        job = runner.start("echo", {"word": "tenny"}, TASKS)
        wait(job)
        self.assertEqual(job.status, "ok")
        self.assertEqual(job.snapshot(0)["lines"], ["hi tenny"])

    def test_unknown_task(self):
        with self.assertRaises(registry.UnknownTask):
            runner.start("nope", {}, TASKS)

    def test_bad_param_starts_nothing(self):
        with self.assertRaises(registry.TaskError):
            runner.start("num", {"n": 99}, TASKS)
        self.assertEqual(runner._jobs, {})

    def test_duplicate_run_rejected(self):
        job = runner.start("sleep", {}, TASKS)
        with self.assertRaises(registry.Busy):
            runner.start("sleep", {}, TASKS)
        self.assertEqual(len(runner._jobs), 1)
        job.stop()

    def test_stop_kills_process(self):
        job = runner.start("sleep", {}, TASKS)
        end = time.time() + 10
        while "go" not in job.snapshot(0)["lines"] and time.time() < end:
            time.sleep(0.05)
        job.stop()
        wait(job)
        self.assertEqual(job.status, "stopped")
        self.assertIsNotNone(job.returncode)

    def test_log_is_capped_and_cursor_is_absolute(self):
        job = runner.start("noisy", {}, TASKS)
        wait(job)
        snap = job.snapshot(0)
        self.assertEqual(snap["next"], 6000)
        self.assertEqual(len(snap["lines"]), runner.MAX_LINES)
        self.assertEqual(snap["lines"][-1], "5999")
        self.assertEqual(job.snapshot(5990)["lines"][0], "5990")
        self.assertEqual(job.snapshot(6000)["lines"], [])

    def test_needs_wsl_missing(self):
        task = {"id": "w", "name": "w", "group": "t", "needs": "wsl", "script": "x.sh"}
        orig = runner.wsl_available
        runner.wsl_available = lambda: False
        try:
            with self.assertRaises(registry.TaskError):
                runner.start("w", {}, {"w": task})
        finally:
            runner.wsl_available = orig

    def test_command_for_script(self):
        task = {"id": "s", "name": "s", "group": "t", "needs": "wsl",
                "script": "tasks/start-sim-wsl.sh", "args": ["--x", "{n}"],
                "params": [{"name": "n", "type": "int", "default": 1}]}
        orig = runner.to_linux_path
        runner.to_linux_path = lambda p: "/linux/" + p.name
        try:
            cmd = runner.build_command(task, {"n": 1})
        finally:
            runner.to_linux_path = orig
        tail = ["/bin/bash", "/linux/start-sim-wsl.sh", "--x", "1"]
        self.assertEqual(cmd[-4:], tail)
        if runner.IS_WIN:
            self.assertEqual(cmd[:4], ["wsl.exe", "-d", runner.DISTRO, "--exec"])

    def test_stdout_closed_after_job_ends(self):
        job = runner.start("echo", {"word": "x"}, TASKS)
        wait(job)
        self.assertTrue(job.proc.stdout.closed)

    @unittest.skipIf(runner.IS_WIN, "POSIX-only test")
    def test_sigterm_ignoring_process_killed(self):
        # Task that ignores SIGTERM and exits on SIGKILL
        task = {"id": "trap", "name": "trap", "group": "t",
                "cmd": [PY, "-u", "-c",
                        "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                        "print('trapped',flush=True);time.sleep(60)"]}
        job = runner.start("trap", {}, {"trap": task})
        end = time.time() + 10
        while "trapped" not in job.snapshot(0)["lines"] and time.time() < end:
            time.sleep(0.05)
        job.stop()
        # Should be stopped within KILL_GRACE + grace for wait
        end = time.time() + (runner.KILL_GRACE + 2)
        while job.status == "running" and time.time() < end:
            time.sleep(0.05)
        self.assertEqual(job.status, "stopped")

    def test_nonexistent_executable_raises_taskerror(self):
        task = {"id": "bad", "name": "bad", "group": "t", "cmd": ["/nonexistent/exe"]}
        with self.assertRaises(registry.TaskError) as cm:
            runner.start("bad", {}, {"bad": task})
        self.assertIn("无法启动", str(cm.exception))
        self.assertEqual(runner._jobs, {})


if __name__ == "__main__":
    unittest.main()
