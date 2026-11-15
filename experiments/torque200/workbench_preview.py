"""Bounded, lazy viewport ownership and one-shot policy synchronization.

The training snapshot protocol has a single request/response slot. Concurrent
opens lease the same pending snapshot; another request cannot overwrite its
files until every waiter has either loaded its own actor or cancelled.
"""
from __future__ import annotations

import json
import re
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from training_control import ControlError


VIEWPORT_PATH = "/api/workbench/v1/viewports"
MAX_VIEWPORTS = 4


class SnapshotCoordinator:
    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.Lock()
        self.request = None
        self.mode = None
        self.leases = set()
        self._status_key = None
        self._status_at = 0.
        self._status_value = None

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def status(self, mode=None):
        # Pending windows share the same runner liveness check; four previews
        # must not multiply supervisor/report validation on every render tick.
        with self.lock:
            now = time.monotonic()
            if (self._status_value is None or mode != self._status_key
                    or now - self._status_at >= .5):
                self._status_value = self.controller.status(mode)
                self._status_key, self._status_at = mode, now
            return dict(self._status_value)

    def request_preview(self, mode):
        with self.lock:
            if self.request is None:
                self.request = self.controller.request_preview(mode)
                self.mode = mode
            elif self.mode != mode:
                raise ControlError("请先关闭其他模式的预览。")
            lease = uuid.uuid4().hex
            self.leases.add(lease)
            return {**self.request, "coordinator_lease": lease}

    def cancel_preview(self, request):
        if not request:
            return
        with self.lock:
            lease = request.get("coordinator_lease")
            if lease not in self.leases:
                return
            self.leases.remove(lease)
            if self.leases:
                return
            released, self.request = self.request, None
            self.mode = None
            self.controller.cancel_preview(released)


class PreviewGroup:
    """At most four simulators, all rendering on one CPU-preview worker.

    A 20-frame/s aggregate budget bounds the additional CPU render work. No ML
    module is imported and no simulator allocated by creating a closed slot.
    """
    def __init__(self, mode, controller, factory, *, initial=None, limit=MAX_VIEWPORTS):
        self.mode, self.factory = mode, factory
        self.controller = SnapshotCoordinator(controller)
        self.limit = min(MAX_VIEWPORTS, max(1, int(limit)))
        self.lock = threading.RLock()
        first = initial or factory(self.controller)
        first.controller = self.controller
        self.previews = {"1": first}
        self.retired = []
        self.closed = False
        self.worker = None
        self.wake = threading.Event()

    @property
    def primary(self):
        with self.lock:
            return next(iter(self.previews.values()))

    @staticmethod
    def _id(value):
        if not isinstance(value, str) or value not in {"1", "2", "3", "4"}:
            raise ValueError("Unknown viewport")
        return value

    def get(self, value=None):
        with self.lock:
            if value is None:
                return next(iter(self.previews.values()))
            identifier = self._id(value)
            if identifier not in self.previews:
                raise ValueError("Unknown viewport")
            return self.previews[identifier]

    def snapshot(self):
        with self.lock:
            records = list(self.previews.items())
        return {"schemaVersion": 1, "limit": self.limit,
                "viewports": [{**preview.snapshot(), "id": identifier,
                               "streamUrl": f"/stream.mjpg?viewport={identifier}"}
                              for identifier, preview in records]}

    def control(self, request):
        action = request.get("action")
        created = None
        with self.lock:
            if self.closed:
                raise ControlError("预览服务正在关闭。")
            if action == "create":
                if len(self.previews) >= self.limit:
                    raise ControlError(f"最多同时保留 {self.limit} 个视窗。")
                created = next(str(index) for index in range(1, self.limit + 1)
                               if str(index) not in self.previews)
                self.previews[created] = self.factory(self.controller)
            else:
                preview = self.get(request.get("id"))
                if action == "remove":
                    if len(self.previews) == 1:
                        preview.suspend()
                    else:
                        identifier = next(key for key, value in self.previews.items() if value is preview)
                        del self.previews[identifier]
                        preview.close()
                        self.retired.append(preview)
                elif action == "open":
                    preview.open()
                elif action == "suspend":
                    preview.suspend()
                elif action in {"pause", "unpause", "reset"}:
                    with preview.lock:
                        if action == "reset":
                            preview.reset_requested = True
                        else:
                            preview.state["paused"] = action == "pause"
                        preview.reset_clock()
                        preview.lock.notify_all()
                else:
                    raise ValueError("Unknown viewport action")
        self.wake.set()
        result = self.snapshot()
        if created is not None:
            result["createdId"] = created
        return result

    def suspend_all(self):
        with self.lock:
            for preview in self.previews.values():
                preview.suspend()
        self.wake.set()

    def tick(self):
        with self.lock:
            records = list(self.previews.values())
            retired, self.retired = self.retired, []
        for preview in retired:
            preview.release()
        visible = sum(not state["suspended"] and state["clients"] > 0
                      for state in (preview.snapshot() for preview in records))
        render_size = (800, 500) if visible <= 1 else (640, 400) if visible == 2 else (400, 250)
        rendered = 0
        for preview in records:
            preview.render_size = render_size
            before = preview.snapshot()["frames"]
            preview.tick()
            rendered += max(0, preview.snapshot()["frames"] - before)
        return rendered

    def run(self):
        try:
            while not self.closed:
                started = time.monotonic()
                rendered = self.tick()
                # Multiple renderers share the original 20 frame/s work budget.
                # Cached paused frames and slots without viewers spend no budget.
                self.wake.wait(timeout=max(.01, .05 * max(1, rendered) - (time.monotonic() - started)))
                self.wake.clear()
        finally:
            with self.lock:
                remaining = list(self.previews.values()) + self.retired
                self.retired = []
            for preview in remaining:
                preview.release()

    def start(self):
        with self.lock:
            if self.worker is not None:
                return
            self.worker = threading.Thread(target=self.run, name="workbench-previews", daemon=True)
            self.worker.start()

    def close(self):
        with self.lock:
            self.closed = True
            for preview in self.previews.values():
                preview.close()
        self.wake.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
        else:
            # Used by deterministic tests and servers which never started the
            # worker. They also own any explicitly ticked test simulators.
            for preview in list(self.previews.values()) + self.retired:
                preview.release()
            self.retired = []


def stream_preview(handler, preview, *, generation=None):
    with preview.lock:
        if generation is not None and generation != preview.generation:
            raise ControlError("此预览连接已过期，请重新打开画面。")
        bound_generation = preview.generation
        if preview.state["clients"] == 0:
            preview.reset_clock()
        preview.state["clients"] += 1
        sequence = preview.state["frames"] - 1  # Send the current cached JPEG to a new viewer.
    try:
        # Validate the generation before committing HTTP headers. Client
        # accounting covers header failures as well as disconnects while
        # writing frames, so a failed handshake cannot leave a phantom viewer.
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        while not preview.closed:
            with preview.lock:
                preview.lock.wait_for(lambda: preview.closed or preview.state["suspended"]
                                      or preview.generation != bound_generation
                                      or preview.state["frames"] != sequence, timeout=1)
                jpg, sequence = preview.jpg, preview.state["frames"]
                if preview.closed or preview.state["suspended"] or preview.generation != bound_generation:
                    break
            if jpg:
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                    str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, TimeoutError):
        pass
    finally:
        with preview.lock:
            preview.state["clients"] -= 1
            if preview.state["clients"] == 0:
                preview.reset_clock()


def dispatch_viewports(handler, group):
    """Handle this narrow route before the generic workbench read API router."""
    try:
        target = urlsplit(handler.path)
    except ValueError:
        handler.send({"error": "Invalid request target"}, 400)
        return True
    stream = target.path == "/stream.mjpg"
    if target.path != VIEWPORT_PATH and not stream:
        return False
    method = getattr(handler, "command", "GET")
    try:
        if stream:
            if method != "GET":
                handler.send({"error": "GET required"}, 405)
                return True
            query = parse_qs(target.query, keep_blank_values=True, strict_parsing=True)
            if set(query) - {"viewport", "generation"} or any(len(values) != 1 for values in query.values()):
                raise ValueError("Invalid viewport query")
            generation = query.get("generation", [None])[0]
            if generation is not None:
                if not re.fullmatch(r"(?:0|[1-9][0-9]{0,15})", generation):
                    raise ValueError("Invalid viewport generation")
                generation = int(generation)
                if generation > 2 ** 53 - 1:
                    raise ValueError("Invalid viewport generation")
            stream_preview(handler, group.get(query.get("viewport", [None])[0]), generation=generation)
        elif target.query:
            raise ValueError("Viewport endpoint does not accept query parameters")
        elif method == "GET":
            handler.send(group.snapshot())
        elif method == "POST":
            if handler.headers.get("Content-Type", "").split(";")[0] != "application/json":
                handler.send({"error": "JSON required"}, 415)
                return True
            size = int(handler.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            request = json.loads(handler.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("Request must be an object")
            if set(request) - {"action", "id"}:
                raise ValueError("Unknown viewport fields")
            handler.send(group.control(request))
        else:
            handler.send({"error": "GET or POST required"}, 405)
    except ControlError as exc:
        handler.send({"error": str(exc)}, 409)
    except (ValueError, TypeError, KeyError) as exc:
        handler.send({"error": str(exc)}, 400)
    return True
