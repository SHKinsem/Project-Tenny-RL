"""Dependency-free, flushed progress logs; a heartbeat is not proof of progress."""
from contextlib import AbstractContextManager
import math
import sys
import threading
import time


class ValidationProgress(AbstractContextManager):
    def __init__(self, heartbeat_seconds=15.0, stream=None):
        if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be finite and positive")
        self.interval = heartbeat_seconds
        self.stream = sys.stdout if stream is None else stream
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.worker = None
        self.current = None
        self.history = []
        self.failed_stage = None
        self.log_error = None

    def _write(self, event, detail=""):
        now = time.monotonic()
        c = self.current
        count = f" | 已完成 {c['completed']}/{c['total']} 项" if c['total'] is not None else ""
        try:
            print(f"[{time.strftime('%H:%M:%S')}] {event} | {c['name']}"
              f" | 阶段 {now-c['started']:.1f}s | 总计 {now-self.started:.1f}s"
              f"{count}{' | '+detail if detail else ''}", file=self.stream, flush=True)
        except Exception as exc:
            # Diagnostic I/O must never replace a model assertion or interrupt.
            self.log_error = type(exc).__name__
            self.stop.set()

    def _finish(self, status):
        if self.current is not None:
            c = self.current
            incomplete = (status == "完成" and c["total"] is not None
                          and c["completed"] != c["total"])
            if incomplete:
                status = "失败"
                self.failed_stage = c["name"]
            self.history.append(dict(stage=c['name'], status=status,
                elapsed_seconds=time.monotonic()-c['started'],
                completed=c['completed'], total=c['total']))
            self._write(status)
            self.current = None
            if incomplete:
                raise ValueError("Cannot finish a stage before all declared checks complete")

    def stage(self, name, *, hint="", total=None):
        """Call only after the previous stage's checks and device waits finish."""
        if total is not None and (not isinstance(total, int) or total < 1):
            raise ValueError("stage total must be a positive integer")
        with self.lock:
            self._finish("完成")
            self.current = dict(name=name, started=time.monotonic(),
                                hint=hint, total=total, completed=0)
            self._write("开始", hint)

    def advance(self):
        """One verified unit has completed, not an estimated compile percentage."""
        with self.lock:
            if self.current is None or self.current['total'] is None:
                raise ValueError("advance requires a stage with an explicit total")
            if self.current['completed'] >= self.current['total']:
                raise ValueError("stage count exceeds declared total")
            self.current['completed'] += 1
            total = self.current["total"]
            done = self.current["completed"]
            if done == total or done % max(1, math.ceil(total / 4)) == 0:
                self._write("进度")

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                if self.current is not None:
                    self._write("等待", "当前阶段尚未返回；心跳不代表检查通过或编译有进展")

    def __enter__(self):
        self.worker = threading.Thread(target=self._heartbeat, name="validation-heartbeat", daemon=True)
        self.worker.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        if self.worker is not None:
            self.worker.join()
        with self.lock:
            if exc_type is not None and self.current is not None:
                self.failed_stage = self.current['name']
            self._finish("失败" if exc_type is not None else "完成")
        return False
