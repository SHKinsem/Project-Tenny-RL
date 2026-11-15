"""Lossless log sink plus a conservative, dependency-free terminal view.

Used only by hybrid-wsl.sh. The shell retains pipefail and the training process's
exit status. Unknown output is shown, not silently classified as harmless.
"""
import argparse
from collections import Counter, deque
import codecs
import json
import math
from pathlib import Path
import re
import sys
import threading
import time

MAX_LINE = 65536
MAX_JSON = 65536
TRAIN_REPORT_KEYS = set("""checkpoint_validated action_mode checkpoint checkpoint_name
checkpoint_sha256 export_sha256 dynamic_adapter_samples requested_steps configured_steps
actual_progress_steps num_envs seed training_including_compile_seconds total_elapsed_seconds
restore_max_abs_error export_max_abs_error adapter_max_abs_error optimizer_resume scope
parameter_leaves_checked parameter_tree_max_abs_error all_parameters_finite
finite_training_losses_checked warm_restart_network_leaves_checked
warm_restart_changed_network_leaves warm_restart_network_max_abs_change
warm_restart_outer_container_before warm_restart_outer_container_after
control_hz ctrl_dt sim_dt physics_hz n_substeps episode_seconds
unroll_length unroll_seconds aggregate_simulated_seconds""".split())
SUCCESS_MARKERS = {"HYBRID_LOCAL_VALIDATION_OK", "HYBRID_TRAIN_RESTORE_EXPORT_OK"}
STAGE = re.compile(r"^\[\d\d:\d\d:\d\d\] (开始|完成|进度|等待|失败) \| (.*?) \| 阶段 ([\d.]+)s \| 总计 ([\d.]+)s(?: \| (.*))?$")
KNOWN_NOTICES = (
    (re.compile(r"E\d{4} .* cuda_executor\.cc:\d+\] Could not get kernel mode driver version: \( INVALID_ARGUMENT: Version does not match the format X\.Y\.Z \)"),
     "驱动版本字符串格式提示", "驱动版本字符串格式提示已记录；不是成功判据，后续检查仍正常执行"),
    (re.compile(r"W\d{4} .* pjrt_executable\.cc:\d+\] Assume version compatibility\. PjRt-IFRT does not track XLA executable versions\."),
     "PjRt 版本提示", "PjRt 版本提示已记录；重复项合并计数，完整内容保留在日志"),
)
LAYER = re.compile(r"hidden_\d+ dict_keys\(\['bias', 'kernel'\]\) \(\d+(?:, \d+)*,?\)")


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class ConsoleView:
    def __init__(self, label, stream=None, *, verbose=False, heartbeat=15.0):
        if not number(heartbeat) or heartbeat <= 0:
            raise ValueError("heartbeat must be finite and positive")
        self.label = label
        self.stream = sys.stdout if stream is None else stream
        self.verbose = verbose
        self.heartbeat = heartbeat
        self.started = self.last_visible = time.monotonic()
        self.stage = "启动 / 等待实际输出"
        self.steps = None
        self.notices = Counter()
        self.pending = []
        self.fragment = ""
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.output_error = None

    def emit(self, text):
        # Called under the same lock by the reader and heartbeat thread.
        print(text, file=self.stream, flush=True)
        self.last_visible = time.monotonic()

    def _known_json(self, value):
        if not isinstance(value, dict):
            return False
        if ({"steps", "elapsed_seconds", "metrics"} <= set(value) <=
                {"steps", "elapsed_seconds", "metrics", "control_hz", "aggregate_simulated_seconds"}):
            metrics = value["metrics"]
            if (type(value["steps"]) is not int or value["steps"] < 0 or
                    not number(value["elapsed_seconds"]) or value["elapsed_seconds"] < 0 or
                    not isinstance(metrics, dict) or not all(number(v) for v in metrics.values()) or
                    any(not number(value[k]) or value[k] < 0 for k in
                        ("control_hz", "aggregate_simulated_seconds") if k in value)):
                return False  # NaN/Inf or changed schemas remain visible in full.
            self.steps = value["steps"]
            self.stage = "训练 / 评估"  # The callback is not a device-completion barrier.
            details = [f"[{self.label}] 实际步数 {self.steps}", f"累计 {value['elapsed_seconds']:.1f}s"]
            if "control_hz" in value:
                details.append(f"控制 {value['control_hz']:g} Hz")
            if "aggregate_simulated_seconds" in value:
                details.append(f"累计各环境仿真 {value['aggregate_simulated_seconds']:.3f}s")
            for key, name in (("eval/episode_reward", "评估回报"), ("training/total_loss", "总损失")):
                if key in metrics:
                    details.append(f"{name} {metrics[key]:.4g}")
            self.emit(" | ".join(details))
            return True
        # This is merely the producer's verbose report, NOT acceptance. The
        # shell checks the command exit and then rereads report.json separately.
        if (value.get("checkpoint_validated") is True and set(value) <= TRAIN_REPORT_KEYS and
                all(k in value for k in ("checkpoint", "actual_progress_steps", "total_elapsed_seconds"))):
            return True
        return False

    def feed(self, text):
        """Frame decoded text without delaying raw file writes for a newline."""
        with self.lock:
            self.fragment += text
            while "\n" in self.fragment:
                line, self.fragment = self.fragment.split("\n", 1)
                self.line(line + "\n")
            if len(self.fragment) >= MAX_LINE or re.search(
                    r"Traceback|(?:Error|Exception):|FATAL|Check failed|LLVM ERROR|Segmentation fault|Aborted|fatal error",
                    self.fragment, re.I):
                line, self.fragment = self.fragment, ""
                self.line(line)

    def line(self, line):
        with self.lock:
            if self.verbose:
                self.emit(line.rstrip("\r\n"))
                return
            text = line.rstrip("\r\n")
            if self.pending:
                # Never buffer an obvious traceback/error behind a partial JSON.
                if (re.search(r"Traceback|(?:Error|Exception):|FATAL|Check failed|^[EWF]\d{4}|^ERROR|^WARNING", text) or
                        not re.match(r'^\s*(?:[{}\[\],"0-9-]|true\b|false\b|null\b|$)', text)):
                    self._flush_pending()
                    self.emit(text)
                    return
                self.pending.append(line)
                joined = "".join(self.pending)
                try:
                    value = json.loads(joined)
                except (ValueError, RecursionError):
                    if len(joined) >= MAX_JSON or len(self.pending) >= 128:
                        self._flush_pending()
                else:
                    if not self._known_json(value):
                        self._flush_pending()
                    else:
                        self.pending.clear()
                return
            if text == "{":
                self.pending = [line]
                return
            if text.startswith("{"):
                try:
                    value = json.loads(text)
                except (ValueError, RecursionError):
                    pass
                else:
                    if self._known_json(value):
                        return
            for pattern, key, explanation in KNOWN_NOTICES:
                if pattern.fullmatch(text):
                    self.notices[key] += 1
                    if self.notices[key] == 1:
                        self.emit(f"[提示] {explanation}")
                    return
            match = STAGE.fullmatch(text)
            if match:
                event, name, elapsed, total, detail = match.groups()
                self.stage = name
                count = re.search(r"已完成 \d+/\d+ 项", detail or "")
                if event in ("开始", "进度", "失败"):
                    self.emit(f"[{self.label}] {event}：{name} | 累计 {total}s" +
                              (f" | {count.group()}" if count else ""))
                # Per-stage timings remain in raw log + stage_timings JSON.
                # Own process emits waits, even if the child holds Python's GIL.
                return
            if text in SUCCESS_MARKERS or LAYER.fullmatch(text):
                return
            if text:
                self.emit(text)  # Unknown warnings, errors, traceback: fail open.

    def _flush_pending(self):
        for line in self.pending:
            self.emit(line.rstrip("\r\n"))
        self.pending.clear()

    def wait_message(self):
        with self.lock:
            if time.monotonic() - self.last_visible >= self.heartbeat:
                # An incomplete JSON or unterminated line must not conceal
                # received diagnostics behind an endless waiting heartbeat.
                self._flush_pending()
                if self.fragment:
                    text, self.fragment = self.fragment, ""
                    self.emit(text.rstrip("\r"))
                self.emit(f"[{self.label}] 等待：{self.stage} | 累计运行 {time.monotonic()-self.started:.0f}s" +
                          (f" | 最后实际步数 {self.steps}" if self.steps is not None else "") +
                          " | 进程尚未返回；心跳不代表取得进展")

    def _heartbeat(self):
        while not self.stop.wait(min(self.heartbeat, 1.0)):
            try:
                self.wait_message()
            except Exception as exc:
                self.output_error = exc
                return

    def start(self):
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join()
        with self.lock:
            if self.output_error is None:
                if self.fragment:
                    text, self.fragment = self.fragment, ""
                    self.line(text)
                self._flush_pending()
                if self.notices:
                    self.emit("[日志] 已合并重复提示：" + "，".join(f"{k} ×{v}" for k, v in self.notices.items()))
        if self.output_error:
            raise self.output_error


def capture(source, logfile, view):
    # Exclusive creation: never truncate an earlier run's evidence. Raw bytes
    # are written BEFORE display decoding/filtering, including invalid UTF-8.
    with Path(logfile).open("xb") as saved:
        view.start()
        decoder = codecs.getincrementaldecoder("utf8")("replace")
        try:
            while chunk := source.read1(MAX_LINE):
                saved.write(chunk)
                saved.flush()
                if view.output_error is None:
                    try:
                        view.feed(decoder.decode(chunk))
                    except Exception as exc:
                        # A broken display must not truncate the producer's log
                        # or inject SIGPIPE into training. Drain to EOF first,
                        # then fail the sink so the shell stops later stages.
                        view.output_error = exc
                        view.stop.set()
        finally:
            if view.output_error is None:
                view.feed(decoder.decode(b"", final=True))
            view.finish()


def summary(label, path, logfile):
    report = json.loads(Path(path).read_text(encoding="utf8"))
    if not isinstance(report, dict) or report.get("error"):
        raise ValueError("报告包含错误或格式不正确")
    if report.get("schema") == "hybrid-local-validation-v1":
        if report.get("runtime_validated") is not True:
            raise ValueError("完整动态预检未通过")
        elapsed = report.get("elapsed_seconds")
        if not number(elapsed) or elapsed < 0:
            raise ValueError("预检耗时缺失或不正确")
        print(f"[通过] {label} | {elapsed:.1f}s | 完整日志：{logfile}", flush=True)
        return
    if report.get("checkpoint_validated") is not True:
        raise ValueError("检查点 / 导出验证未通过")
    values = [report.get(k) for k in ("actual_progress_steps", "requested_steps", "configured_steps")]
    if not all(type(v) is int and v > 0 for v in values) or not (values[0] >= values[2] >= values[1]):
        raise ValueError("训练步数报告缺失或未完成")
    actual, requested, configured = values
    elapsed, training = (report.get(k) for k in ("total_elapsed_seconds", "training_including_compile_seconds"))
    if not all(number(v) and v >= 0 for v in (elapsed, training)):
        raise ValueError("训练耗时缺失或不正确")
    print(f"[通过] {label} | 实际 {actual} 步（请求 {requested} / 配置 {configured}）"
          f" | 编译+训练 {training:.1f}s | 总耗时 {elapsed:.1f}s", flush=True)
    print(f"[产物] 保存 / 重载 / 导出校验通过 | 完整日志：{logfile}", flush=True)
    if "warm_restart_changed_network_leaves" in report:
        print("[热启动] 网络和归一化恢复，优化器重新初始化；不是精确续训。", flush=True)


def failure(label, logfile, code):
    print(f"[失败] {label} | 退出码 {code} | 完整日志：{logfile}", file=sys.stderr, flush=True)
    path = Path(logfile)
    if path.exists():
        with path.open("rb") as stream:
            # Bounded tail avoids dumping megabytes of metrics on failure.
            stream.seek(max(0, path.stat().st_size - 8192))
            tail = deque(stream.read().decode("utf8", "replace").splitlines(), maxlen=12)
        print("[最近日志片段，完整错误请见上述日志]", file=sys.stderr)
        for line in tail:
            print(line[:500] + (" …" if len(line) > 500 else ""), file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("capture", "summary", "failure"))
    p.add_argument("--label", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--report")
    p.add_argument("--code", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--heartbeat", type=float, default=15.0)
    args = p.parse_args()
    try:
        if args.action == "capture":
            capture(sys.stdin.buffer, args.log, ConsoleView(args.label, verbose=args.verbose, heartbeat=args.heartbeat))
        elif args.action == "summary":
            summary(args.label, args.report, args.log)
        else:
            failure(args.label, args.log, args.code)
    except Exception as exc:
        print(f"[日志处理失败] {type(exc).__name__}: {exc} | 日志：{args.log}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
