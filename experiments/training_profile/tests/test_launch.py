"""CPU-only checks for cancellation of an isolated diagnostic process."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launch import stop_process_group


@unittest.skipUnless(os.name == "posix", "Process-group launch is a WSL/Linux feature")
class CancellationTests(unittest.TestCase):
    def run_child(self, handlers):
        code = ("import signal,time\n" + handlers +
                "\nprint('ready', flush=True)\ntime.sleep(60)\n")
        child = subprocess.Popen([sys.executable, "-u", "-c", code],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True, text=True)
        self.addCleanup(child.stdout.close)

        def cleanup():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        self.addCleanup(cleanup)
        self.assertEqual(child.stdout.readline().strip(), "ready")
        return child

    def test_interrupt_exits_and_reaps_child(self):
        child = self.run_child("")
        stop_process_group(child, interrupt_seconds=.1, terminate_seconds=.1)
        self.assertIsNotNone(child.returncode)

    def test_native_work_ignoring_interrupt_and_terminate_is_killed(self):
        child = self.run_child("signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                               "signal.signal(signal.SIGTERM, signal.SIG_IGN)")
        started = time.monotonic()
        stop_process_group(child, interrupt_seconds=.1, terminate_seconds=.1)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertLess(time.monotonic() - started, 5)


if __name__ == "__main__":
    unittest.main()
