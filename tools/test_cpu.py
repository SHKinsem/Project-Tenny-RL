"""Run unittest with UTF-8 children and explicit Git Bash for Windows shell tests.

Run from the repository root; arguments are forwarded to unittest unchanged.
See docs/validation.md for test selection and loopback HTTP permissions.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest


def main():
    os.environ["PYTHONUTF8"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    original_run = subprocess.run

    def run(argv, *args, **kwargs):
        if (os.name == "nt" and isinstance(argv, (list, tuple)) and argv
                and argv[0] in ("bash", "bash.exe")):
            configured = os.environ.get("TENNY_TEST_BASH")
            bash = (Path(configured) if configured else
                    Path(os.environ["ProgramFiles"]) / "Git" / "bin" / "bash.exe")
            if not bash.is_absolute() or not bash.is_file():
                raise FileNotFoundError("Set TENNY_TEST_BASH to the absolute Git Bash executable path")
            argv = [str(bash), *argv[1:]]
        return original_run(argv, *args, **kwargs)

    # Existing shell tests use subprocess.run(['bash', ...]). Windows resolves
    # that bare name to the WSL launcher even when PATH lists Git Bash first.
    subprocess.run = run
    try:
        unittest.main(module=None)
    finally:
        subprocess.run = original_run


if __name__ == "__main__":
    main()
