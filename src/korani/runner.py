"""Subprocess execution of generated simulation scripts (stage E).

Each script runs with the same interpreter in its own working directory with
a wall-clock timeout; stdout/stderr are captured to log files next to the
script so failed runs stay diagnosable. The generated code is saved for
human review before and after every run — nothing executes hidden.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_TAIL_CHARS = 4000


@dataclass
class RunResult:
    exit_code: Optional[int]  # None if timed out
    timed_out: bool
    duration_s: float
    stdout_path: str
    stderr_path: str
    stderr_tail: str  # what the Debugger sees


def run_script(script_path: str, cwd: str, timeout_s: float = 900.0) -> RunResult:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"  # generated code must never open a GUI
    script = Path(script_path).resolve()
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )
        exit_code, timed_out = proc.returncode, False
        out, err = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code, timed_out = None, True
        out = exc.stdout or b""
        err = exc.stderr or b""
    duration = time.monotonic() - start

    cwd_path = Path(cwd)
    stdout_path = cwd_path / "run_stdout.log"
    stderr_path = cwd_path / "run_stderr.log"
    stdout_path.write_bytes(out)
    stderr_path.write_bytes(err)

    stderr_tail = err.decode("utf-8", errors="replace")[-_TAIL_CHARS:]
    if timed_out:
        stderr_tail = (
            "TIMEOUT: the script exceeded the %.0f s wall-clock limit.\n%s"
            % (timeout_s, stderr_tail)
        )
    return RunResult(
        exit_code=exit_code,
        timed_out=timed_out,
        duration_s=duration,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stderr_tail=stderr_tail,
    )
