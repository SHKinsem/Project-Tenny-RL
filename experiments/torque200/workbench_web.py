"""Serve the real workbench from the existing console without replacing control."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

if __package__:
    from .workbench_api import WorkbenchError, WorkbenchReader
    from .workbench_api.http import PREFIX, read_request
    from .workbench_api.mode_catalog import MODES, mode_spec
else:
    from workbench_api import WorkbenchError, WorkbenchReader
    from workbench_api.http import PREFIX, read_request
    from workbench_api.mode_catalog import MODES, mode_spec


UI_ROOT = Path(__file__).with_name("workbench-ui")
ASSETS = frozenset({"app.mjs", "adapter.mjs", "transport.mjs", "layout.mjs", "chart-model.mjs",
                    "chart-view.mjs", "styles.css", "tokens.css"})
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
       "connect-src 'self'; img-src 'self' data: blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'self'")


def mode_catalog(mode):
    """Presentation names belong to this backend adapter, not to the UI or PPO."""
    names = {"original": "腿位置＋轮速度", "torque200": "全扭矩", "wheel_torque_bc": "腿位置＋轮扭矩",
             "leg_force_bc": "腿力＋轮扭矩"}
    actions = {"virtual_leg_force": "腿力＋轮扭矩", "matched_envelope": "匹配包络", "direct_torque": "直接扭矩"}
    modes = []
    for key in MODES:
        spec = mode_spec(key)
        modes.append({"id": key, "label": names.get(key, actions.get(spec["action_mode"], key)),
                      "controlHz": spec["control_hz"], "obsDim": spec["obs_dim"], "actionMode": spec["action_mode"],
                      "teacher": bool(spec.get("teacher")), "hybrid": spec["hybrid"]})
    return {"schemaVersion": 1, "modes": modes, "currentMode": mode}


def _send(handler, body, status=200, content_type="application/json; charset=utf-8", *, policy=False, allow=False):
    data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    if policy:
        handler.send_header("Content-Security-Policy", CSP)
    if allow:
        handler.send_header("Allow", "GET, HEAD")
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(data)


def _error(handler, code, message, status, *, allow=False):
    _send(handler, {"schemaVersion": 1, "error": {"code": code, "message": message}}, status, allow=allow)


def _query(target, allowed):
    values = parse_qs(urlsplit(target).query, keep_blank_values=True, strict_parsing=True, max_num_fields=8)
    if set(values) - set(allowed) or any(len(value) != 1 for value in values.values()):
        raise ValueError("Unsupported or repeated query parameter")
    return {key: value[0] for key, value in values.items()}


def _body(handler, allowed):
    host = handler.headers.get("Host", "")
    if handler.headers.get("Origin") not in (None, "http://" + host):
        raise WorkbenchError("same_origin_required", 403, "Same-origin requests required")
    if handler.headers.get("Content-Type", "").split(";")[0] != "application/json":
        raise WorkbenchError("json_required", 415, "JSON required")
    try:
        size = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        raise ValueError("Invalid request size") from None
    if not 0 < size <= 65536 or handler.headers.get("Transfer-Encoding"):
        raise ValueError("Invalid request size")
    data = handler.rfile.read(size)
    if len(data) != size:
        raise ValueError("Incomplete request body")
    def reject(_):
        raise ValueError("Nonfinite JSON is not accepted")
    value = json.loads(data, parse_constant=reject)
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("Unsupported request fields")
    return value


def dispatch(handler, *, artifacts, mode, controller=None):
    """Return true when handled; leave legacy state/commands/stream untouched."""
    try:
        path = urlsplit(handler.path).path
    except ValueError:
        _error(handler, "invalid_request", "Invalid request target", 400)
        return True
    owned = path in {"/", "/legacy", "/health/workbench"} or path.startswith(("/workbench/", PREFIX))
    if not owned:
        return False
    control_paths = {PREFIX + name for name in ("storage", "storage/resolve", "import")}
    if handler.command not in {"GET", "HEAD"} and not (controller is not None and handler.command == "POST" and path in control_paths):
        _error(handler, "read_only", "This endpoint only supports GET and HEAD", 405, allow=True)
        return True
    try:
        if controller is not None and path in control_paths and handler.command == "POST":
            if urlsplit(handler.path).query:
                raise ValueError("This endpoint does not accept query parameters")
            request = _body(handler, {"path", "mode"} if path == PREFIX + "import" else {"path"})
            if path == PREFIX + "storage":
                result = controller.configure_storage(request.get("path"))
            elif path == PREFIX + "storage/resolve":
                result = controller.resolve_directory(request.get("path"))
            else:
                imported = controller.import_checkpoint(request.get("path"), request.get("mode", mode))
                model = imported.get("model")
                result = {**imported, "checkpoint": controller._checkpoint_option(model) if model else imported.get("checkpoint"),
                          "storage": controller.storage_state()}
            _send(handler, result)
            return True
        if path.startswith(PREFIX):
            if controller is not None and path in {PREFIX + "parameters", PREFIX + "checkpoint-options", PREFIX + "storage"}:
                allowed = {"mode", "checkpoint"} if path == PREFIX + "parameters" else ({"mode", "limit"} if path == PREFIX + "checkpoint-options" else set())
                query = _query(handler.path, allowed)
                selected_mode = query.get("mode", mode)
                if path == PREFIX + "parameters":
                    result = controller.parameters(selected_mode, checkpoint=query.get("checkpoint"))
                elif path == PREFIX + "checkpoint-options":
                    result = controller.checkpoint_options(selected_mode, limit=int(query.get("limit", "50")))
                else:
                    result = controller.storage_state()
                _send(handler, result)
                return True
            if path == PREFIX + "catalog":
                if urlsplit(handler.path).query:
                    _error(handler, "invalid_query", "Catalog does not accept query parameters", 400)
                else:
                    _send(handler, mode_catalog(mode))
                return True
            # The standalone read-only capability document is mode-independent;
            # the integrated supervisor accepts a mode for live control rights.
            target = PREFIX + "capabilities" if controller is not None and path == PREFIX + "capabilities" else handler.path
            result = read_request(WorkbenchReader(artifacts), target, mode=mode)
            if path == PREFIX + "capabilities":
                result = {**result, "scope": "workbench_read_api", "legacyControls": {
                    "startTraining": True, "resumeLatest": True, "stopAndSave": True,
                    "preview": True, "switchMode": True, "pauseTraining": False,
                    "saveSnapshot": False, "resumeSelected": False, "parameterWrites": False,
                    "multipleViewports": False, "directoryBridge": False}}
                if controller is not None and hasattr(controller, "capabilities"):
                    query = _query(handler.path, {"mode"})
                    result["legacyControls"] = controller.capabilities(query.get("mode", mode))
                    result["scope"] = "workbench"
            _send(handler, result)
            return True
        if path == "/health/workbench":
            _send(handler, {"workbench": True, "mode": mode, "mock": False})
            return True
        if path == "/legacy":
            data = Path(__file__).with_name("dashboard.html").read_bytes()
            _send(handler, data, content_type="text/html; charset=utf-8")
            return True
        if path == "/":
            filename = "index.html"
        else:
            filename = path.removeprefix("/workbench/")
            if filename not in ASSETS:
                _error(handler, "not_found", "Unknown workbench asset", 404)
                return True
        root = UI_ROOT.resolve()
        target = UI_ROOT / filename
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            _error(handler, "not_found", "Unknown workbench asset", 404)
            return True
        if target.stat().st_size > 2 * 1024 * 1024:
            _error(handler, "resource_too_large", "Workbench asset exceeds its size limit", 413)
            return True
        content_type = {".html": "text/html; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
                        ".css": "text/css; charset=utf-8"}[target.suffix]
        _send(handler, target.read_bytes(), content_type=content_type, policy=filename == "index.html")
        return True
    except WorkbenchError as exc:
        _error(handler, exc.code, exc.message, exc.status)
    except (ValueError, TypeError, KeyError) as exc:
        _error(handler, "invalid_request", str(exc), 400)
    except RuntimeError as exc:
        _error(handler, "control_conflict", str(exc), 409)
    except OSError:
        _error(handler, "workbench_unavailable", "Workbench files are unavailable", 503)
    return True
