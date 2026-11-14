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


if __name__ == "__main__":
    unittest.main()
