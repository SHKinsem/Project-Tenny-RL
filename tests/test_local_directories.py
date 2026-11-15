"""Directory UI stays mocked; mapping, validation and cancellation are real."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tenny.local_directories import (DirectoryError, LocalDirectories,
                                     backend_to_windows, windows_to_backend)


class LocalDirectoryTests(unittest.TestCase):
    def test_local_drive_and_current_distribution_map_in_both_directions(self):
        for windows, linux in ((r"C:\Models\run 01", "/mnt/c/Models/run 01"),
                               (r"\\wsl.localhost\Ubuntu\home\robot\runs", "/home/robot/runs"),
                               (r"\\wsl$\Ubuntu\home\robot\runs", "/home/robot/runs")):
            with self.subTest(path=windows):
                self.assertEqual(windows_to_backend(windows, "Ubuntu"), linux)
                mapped = backend_to_windows(linux, "Ubuntu")
                self.assertEqual(windows_to_backend(mapped, "Ubuntu"), linux)

    def test_network_device_relative_parent_and_foreign_distribution_paths_fail(self):
        for windows in (r"relative\folder", r"C:\Models\..\private", r"\\server\share\path",
                        r"\\wsl.localhost\OtherDistro\home", r"\\?\C:\device"):
            with self.subTest(path=windows), self.assertRaises(DirectoryError):
                windows_to_backend(windows, "Ubuntu")
        for linux in ("relative", "/home/../private", "/home/robot\\escaped"):
            with self.subTest(path=linux), self.assertRaises(DirectoryError):
                backend_to_windows(linux, "Ubuntu")

    def test_selection_returns_cancelled_without_configuring_anything(self):
        picker = Mock(return_value=None)
        directories = LocalDirectories(platform="win32", distro="Ubuntu", picker=picker)
        result = directories.select(purpose="save", initial="/home/robot/models")
        self.assertEqual(result, {"cancelled": True, "path": None, "displayPath": None})
        picker.assert_called_once_with("save", r"\\wsl.localhost\Ubuntu\home\robot\models")

    def test_target_distribution_is_verified_without_default_wsl_probe(self):
        directories = LocalDirectories(platform="win32", picker=lambda *_: None)
        with patch.dict("os.environ", {}, clear=True), patch("tenny.local_directories.subprocess.run") as run:
            self.assertFalse(directories.capabilities()["available"])
            directories.bind_target_distro("Ubuntu-production")
            self.assertTrue(directories.capabilities()["available"])
            self.assertTrue(directories.capabilities()["distroVerified"])
            self.assertEqual(directories.distro, "Ubuntu-production")
            run.assert_not_called()
        with self.assertRaises(DirectoryError):
            directories.bind_target_distro("WrongDistro")

    def test_selection_paths_remain_data_and_open_only_uses_validated_folder(self):
        path = r"C:\Models\literal $(value) & ; data"
        opener = Mock()
        directories = LocalDirectories(platform="win32", distro="Ubuntu", picker=lambda *_: path, opener=opener)
        result = directories.select(purpose="import")
        self.assertEqual(result["path"], "/mnt/c/Models/literal $(value) & ; data")
        directories.open(result["path"])
        opener.assert_called_once_with(path)

    def test_unsupported_platform_and_concurrent_selection_are_explicit(self):
        directories = LocalDirectories(platform="linux", distro="Ubuntu")
        self.assertFalse(directories.capabilities()["available"])
        with self.assertRaises(DirectoryError):
            directories.select(purpose="save")
        directories = LocalDirectories(platform="win32", distro="Ubuntu", picker=lambda *_: None)
        directories._selection_lock.acquire()
        self.addCleanup(directories._selection_lock.release)
        with self.assertRaises(DirectoryError):
            directories.select(purpose="save")

    def test_helper_launch_uses_argv_and_never_a_shell(self):
        completed = Mock(stdout='{"path":null}')
        with patch("tenny.local_directories.subprocess.run", return_value=completed) as run:
            self.assertIsNone(LocalDirectories._pick("save", r"C:\Models\$(literal)"))
        argv = run.call_args.args[0]
        self.assertEqual(argv[-1], r"C:\Models\$(literal)")
        self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
