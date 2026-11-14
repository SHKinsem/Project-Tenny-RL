"""任务清单：加载 tasks.toml、校验参数、生成 argv。"""
import math
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOML = Path(__file__).with_name("tasks.toml")


class TaskError(ValueError):
    pass


class UnknownTask(TaskError):
    pass


class Busy(TaskError):
    pass


def load(path=TOML):
    with open(path, "rb") as f:
        tasks = tomllib.load(f)["task"]
    out = {}
    for t in tasks:
        if t["id"] in out:
            raise TaskError(f"重复的任务 id: {t['id']}")
        if ("cmd" in t) == ("script" in t):
            raise TaskError(f"{t['id']}: cmd 与 script 必须二选一")
        out[t["id"]] = t
    return out


def _coerce(p, raw):
    name, kind = p["name"], p["type"]
    if kind in ("int", "float"):
        try:
            v = int(raw) if kind == "int" else float(raw)
        except (TypeError, ValueError, OverflowError):
            raise TaskError(f"{name}: 需要{kind}")
        if not math.isfinite(v):
            raise TaskError(f"{name}: 需要有限数值")
        if v < p.get("min", v) or v > p.get("max", v):
            raise TaskError(f"{name}: 超出范围 [{p.get('min')}, {p.get('max')}]")
        return v
    if kind == "choice":
        if str(raw) not in p["choices"]:
            raise TaskError(f"{name}: 必须是 {p['choices']} 之一")
        return str(raw)
    if kind == "flag":
        return raw in (True, "true", "1", 1, "on")
    if kind == "str":
        return str(raw)
    raise TaskError(f"{name}: 未知参数类型 {kind}")


def resolve_params(task, given):
    given = given or {}
    params = task.get("params", [])
    unknown = set(given) - {p["name"] for p in params}
    if unknown:
        raise TaskError(f"未登记的参数: {sorted(unknown)}")
    return {p["name"]: _coerce(p, given.get(p["name"], p.get("default"))) for p in params}


def build_args(task, values):
    out = [a.format(**values) for a in task.get("args", [])]
    out += [p["flag"] for p in task.get("params", []) if p["type"] == "flag" and values[p["name"]]]
    return out
