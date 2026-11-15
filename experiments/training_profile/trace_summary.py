"""Summarize measured GPU-stream events from a JAX Chrome trace.

Chrome ``ts`` and ``dur`` values are microseconds; ``displayTimeUnit`` only
controls a viewer. Device-stream metadata is required. Host executor and CUDA
submission events are never substituted for measured device execution.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


_UNKNOWN = frozenset({"unknown", "mixed"})


def _read_text(path):
    path = Path(path)
    with (gzip.open(path, "rt", encoding="utf8") if path.suffix == ".gz" else
          path.open("r", encoding="utf8")) as stream:
        return stream.read()


def _quoted_values(line, name):
    result = []
    for value in re.findall(r"\b" + re.escape(name) + r'\s*=\s*"((?:\\.|[^"\\])*)"', line):
        try:
            result.append(json.loads('"' + value + '"'))
        except ValueError:
            result.append(value)
    return result


def _parse_hlo(text):
    module = re.search(r"^HloModule\s+([^,\s]+)", text)
    if module is None:
        raise ValueError("Expected textual HloModule")
    computations, instructions, current, entry = {}, defaultdict(list), None, None
    for line in text.splitlines():
        header = re.match(r"^(ENTRY\s+)?%?([\w.-]+)\s*(?:\(|\{).*\{\s*$", line)
        if header:
            current = header.group(2)
            computations[current] = []
            if header.group(1):
                entry = current
            continue
        if line.strip() == "}":
            current = None
            continue
        match = re.match(r"\s*(?:ROOT\s+)?%?([\w.-]+)\s*=\s*(.*)", line)
        if current is None or match is None:
            continue
        name, rhs = match.groups()
        refs = re.findall(r"\b(?:calls|to_apply|body|condition|true_computation|false_computation)\s*=\s*%?([\w.-]+)", rhs)
        for group in re.findall(r"\bcalled_computations\s*=\s*\{([^}]*)\}", rhs):
            refs.extend(re.findall(r"%?([\w.-]+)", group))
        source = _quoted_values(rhs, "source_file")
        source_line = re.search(r"\bsource_line\s*=\s*(\d+)", rhs)
        instruction = {"name": name, "computation": current, "op_names": _quoted_values(rhs, "op_name"),
                       "references": list(dict.fromkeys(refs)), "aliases": _quoted_values(rhs, "deduplicated_name"),
                       "source_file": source[0] if source else None,
                       "source_line": int(source_line.group(1)) if source_line else None}
        computations[current].append(instruction)
        instructions[name].append(instruction)
    return {"module": module.group(1), "entry": entry, "computations": computations,
            "instructions": dict(instructions), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def _tags(text, prefix):
    return set(re.findall(re.escape(prefix) + r"[A-Za-z0-9_.-]+", text))


def _without_stages(text, prefix):
    # Removing a named scope inside jvp(ppo.foo) deliberately leaves jvp(),
    # matching the corresponding unannotated JAX metadata rather than guessing.
    text = re.sub(re.escape(prefix) + r"[A-Za-z0-9_.-]+", "", text)
    return "/".join(segment for segment in text.split("/") if segment)


def _join_scope(prefix, local):
    if not prefix:
        return {local}
    if not local:
        return {prefix}
    left, right = prefix.split("/"), local.split("/")
    result = {prefix + "/" + local}
    for size in range(min(len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            result.add("/".join(left + right[size:]))
            break
    return result


def _annotation_index(annotated, targets, prefix):
    """Recover full op_name paths by following actual unoptimized HLO calls.

    Only paths whose scope-stripped form exactly equals a compiled metadata
    op_name enter the index. Prefix checks prune impossible calling contexts;
    duplicate exact keys retain every observed stage set for ambiguity checks.
    """
    prefixes = set()
    for name in targets:
        segments = name.split("/")
        prefixes.update("/".join(segments[:end]) for end in range(1, len(segments) + 1))
    index, seen = defaultdict(set), set()
    details = {"exact_op_names_matched": 0, "calling_contexts": 0, "context_limit_reached": False,
               "missing_computations": 0, "recursive_calls": 0}

    def visit(computation, context, ancestors):
        if computation in ancestors:
            details["recursive_calls"] += 1
            return
        if (computation, context) in seen:
            return
        if len(seen) >= 100_000:
            details["context_limit_reached"] = True
            return
        seen.add((computation, context))
        if computation not in annotated["computations"]:
            details["missing_computations"] += 1
            return
        for instruction in annotated["computations"][computation]:
            paths = set()
            for local in instruction["op_names"] or [""]:
                paths.update(_join_scope(context, local))
            viable = {name for name in paths if _without_stages(name, prefix) in prefixes}
            for name in viable:
                key = _without_stages(name, prefix)
                if key in targets:
                    index[key].add(tuple(sorted(_tags(name, prefix))))
                for reference in instruction["references"]:
                    visit(reference, name, ancestors | {computation})
    if annotated["entry"]:
        visit(annotated["entry"], "", set())
    details.update(exact_op_names_matched=len(index), calling_contexts=len(seen),
                   exact_op_names_with_stage_labels=sum(any(tags for tags in values) for values in index.values()),
                   ambiguous_exact_op_names=sum(len(value) > 1 for value in index.values()))
    return dict(index), details


def _consensus(categories):
    categories = set(categories)
    if not categories:
        return "unknown"
    if len(categories) == 1:
        return next(iter(categories))
    domains = {category.split(".", 1)[0] for category in categories}
    if len(domains) == 1 and next(iter(domains)) in ("rollout", "update", "physics", "ppo"):
        return next(iter(domains)) + ".mixed"
    return "mixed"


def _source_subsystem(name):
    """Only explicit function-path evidence, never a CUDA kernel-name guess."""
    if "/step/forward/solve/" in name or name.endswith("/step/forward/solve"):
        return "physics.constraint_solver"
    if "/step/" in name and ("/collision/" in name or name.endswith("/collision")):
        return "physics.collision"
    if "/step/forward/" in name:
        return "physics.forward_other"
    if "/compute_gae/" in name or name.endswith("/compute_gae"):
        return "ppo.gae"
    if "/compute_ppo_loss/" in name or name.endswith("/compute_ppo_loss"):
        return "ppo.loss_combined"
    return "unknown"


def _subsystem_index(compiled):
    direct = {}
    for records in compiled["computations"].values():
        for record in records:
            names = [name for name in record["op_names"] if _source_subsystem(name) != "unknown"]
            direct[id(record)] = (_consensus(_source_subsystem(name) for name in names), names[:2])
    contexts, seen = defaultdict(list), set()
    pending = [(compiled["entry"], ("unknown", []))] if compiled["entry"] else []
    while pending and len(seen) < 100_000:
        computation, context = pending.pop()
        if (computation, context[0]) in seen:
            continue
        seen.add((computation, context[0]))
        contexts[computation].append(context)
        for record in compiled["computations"].get(computation, ()):
            own = direct[id(record)]
            pending.extend((reference, own if own[0] != "unknown" else context)
                           for reference in record["references"])
    cache, active = {}, set()

    def computation_info(name):
        if name in cache:
            return cache[name]
        if name in active:
            return "unknown", []
        active.add(name)
        values = [instruction_info(record) for record in compiled["computations"].get(name, ())]
        active.remove(name)
        known = [value for value in values if value[0] != "unknown"]
        value = (_consensus(item[0] for item in known),
                 list(dict.fromkeys(sample for item in known for sample in item[1]))[:2])
        cache[name] = value
        return value

    def instruction_info(record):
        own = direct[id(record)]
        if own[0] != "unknown":
            return own
        values = [computation_info(name) for name in record["references"]]
        known = [value for value in values if value[0] != "unknown"]
        if not known:
            known = contexts.get(record["computation"], ())
        return (_consensus(value[0] for value in known),
                list(dict.fromkeys(sample for value in known for sample in value[1]))[:2])
    return {name: (_consensus(instruction_info(record)[0] for record in records),
                   list(dict.fromkeys(sample for record in records for sample in instruction_info(record)[1]))[:2])
            for name, records in compiled["instructions"].items()}


def _classify_tags(tags, prefix):
    category = _classify({"args": {"name": "/".join(tags)}}, prefix)[0]
    if category == "mixed":
        domains = {_classify({"args": {"name": tag}}, prefix)[0].split(".", 1)[0] for tag in tags}
        if len(domains) == 1 and next(iter(domains)) in ("rollout", "update"):
            return next(iter(domains)) + ".mixed"
    return category


def _hlo_index(compiled_hlo_path, annotated_mlir_path, prefix, source_stage_ranges):
    compiled = _parse_hlo(_read_text(compiled_hlo_path))
    targets = {name for records in compiled["instructions"].values() for record in records
               for name in record["op_names"]}
    annotations, annotation_details = {}, None
    if annotated_mlir_path is not None:
        text = _read_text(annotated_mlir_path)
        if text.lstrip().startswith("HloModule"):
            annotated = _parse_hlo(text)
            conversion = "textual_hlo"
        else:
            # This conversion performs no device compilation or execution.
            try:
                from jaxlib import xla_client
            except ImportError as exc:
                raise RuntimeError("annotated_mlir_path requires the profiling interpreter's jaxlib for CPU conversion") from exc
            hlo = xla_client._xla.mlir.mlir_module_to_xla_computation(text, False, False).as_hlo_module()
            options = xla_client._xla.HloPrintOptions()
            options.print_metadata = True
            annotated = _parse_hlo(hlo.to_string(options))
            conversion = "jaxlib_cpu_mlir_to_hlo"
        if annotated["module"] != compiled["module"]:
            raise ValueError("Annotated and compiled HLO module names differ")
        annotations, annotation_details = _annotation_index(annotated, targets, prefix)
        annotation_details.update(conversion=conversion, hlo_text_sha256=annotated["sha256"],
                                  input_text_sha256=hashlib.sha256(text.encode()).hexdigest())
    rules = source_stage_ranges or []
    for rule in rules:
        if (not isinstance(rule, dict) or not isinstance(rule.get("path_suffix"), str) or
                not isinstance(rule.get("stage"), str) or not rule["stage"].startswith(prefix)):
            raise ValueError("source_stage_ranges need path_suffix and a stage beginning with stage_prefix")
    resolved, recursion = {}, set()

    def direct_uncached(record):
        candidates, tags, origins, samples = [], set(), [], []
        for name in record["op_names"]:
            own = _tags(name, prefix)
            if own:
                candidates.append(_classify_tags(own, prefix))
                tags.update(own)
                origins.append("compiled_op_name")
                samples.append(name)
            elif name in annotations:
                for found in annotations[name]:
                    candidates.append(_classify_tags(found, prefix))
                    tags.update(found)
                origins.append("annotated_exact_op_name")
                samples.append(name)
        if candidates:
            return _consensus(candidates), tags, origins, samples[:2]
        for rule in rules:
            file = (record["source_file"] or "").replace("\\", "/")
            line = record["source_line"]
            if (file.endswith(rule["path_suffix"].replace("\\", "/")) and
                    ("start_line" not in rule or line is not None and line >= rule["start_line"]) and
                    ("end_line" not in rule or line is not None and line <= rule["end_line"])):
                tags.add(rule["stage"])
                candidates.append(_classify({"args": {"name": rule["stage"]}}, prefix)[0])
                origins.append("explicit_source_range")
                samples.append(file + ":" + str(line))
        return (_consensus(candidates), tags, origins, samples[:2]) if candidates else None

    direct_cache = {id(record): direct_uncached(record)
                    for records in compiled["computations"].values() for record in records}

    def direct(record):
        return direct_cache[id(record)]

    # An unlabelled while-condition/fusion can inherit an explicitly labelled
    # callsite. Every calling context is retained; cross-domain reuse stays mixed.
    inherited, seen_contexts = defaultdict(list), set()
    pending = [(compiled["entry"], ("unknown", set(), ["entry_unclassified"], []))] if compiled["entry"] else []
    while pending and len(seen_contexts) < 100_000:
        computation, context = pending.pop()
        key = computation, context[0], tuple(sorted(context[1]))
        if key in seen_contexts:
            continue
        seen_contexts.add(key)
        inherited[computation].append(context)
        for record in compiled["computations"].get(computation, ()):
            own = direct(record)
            scope = own if own is not None and own[1] else context
            scope = scope[0], scope[1], list(dict.fromkeys(["inherited_callsite", *scope[2]])), scope[3]
            pending.extend((reference, scope) for reference in record["references"])

    def inherited_info(computation):
        values = inherited.get(computation, ())
        return (_consensus(value[0] for value in values),
                set().union(*(value[1] for value in values)) if values else set(),
                list(dict.fromkeys(origin for value in values for origin in value[2])),
                list(dict.fromkeys(sample for value in values for sample in value[3]))[:2])

    def computation_info(name):
        if name in resolved:
            return resolved[name]
        if name in recursion:
            return "unknown", set(), ["recursive_call"], []
        recursion.add(name)
        values = [instruction_info(record) for record in compiled["computations"].get(name, ())]
        recursion.remove(name)
        # Unlabelled constants and parameters do not override labelled work.
        known = [value for value in values if value[0] != "unknown"]
        result = (_consensus(value[0] for value in known),
                  set().union(*(value[1] for value in known)) if known else set(),
                  list(dict.fromkeys(origin for value in known for origin in value[2])),
                  list(dict.fromkeys(sample for value in known for sample in value[3]))[:2])
        resolved[name] = result
        return result

    def instruction_info(record):
        own = direct(record)
        if own is not None and (own[0] != "unknown" or own[1]):
            return own
        values = [computation_info(name) for name in record["references"]]
        known = [value for value in values if value[0] != "unknown"]
        if not known:
            return inherited_info(record["computation"])
        return (_consensus(value[0] for value in known),
                set().union(*(value[1] for value in known)) if known else set(),
                list(dict.fromkeys(["called_computation", *(origin for value in known for origin in value[2])])) if known else [],
                list(dict.fromkeys(sample for value in known for sample in value[3]))[:2])

    exact, aliases = {}, defaultdict(list)
    subsystems = _subsystem_index(compiled)
    for name, records in compiled["instructions"].items():
        values = [instruction_info(record) for record in records]
        value = {"category": _consensus(item[0] for item in values),
                 "tags": sorted(set().union(*(item[1] for item in values))),
                 "origins": sorted({origin for item in values for origin in item[2]}),
                 "op_names": sorted({op for item in values for op in item[3]})[:2],
                 "subsystem": subsystems[name][0], "subsystem_sources": subsystems[name][1]}
        exact[name] = value
        for record in records:
            for alias in [name, *record["aliases"]]:
                aliases[re.sub(r"[^A-Za-z0-9_]", "_", alias)].append(value)
    details = {"module_name": compiled["module"], "hlo_text_sha256": compiled["sha256"],
               "computation_count": len(compiled["computations"]),
               "instruction_count": sum(map(len, compiled["instructions"].values())),
               "compiled_unique_op_names": len(targets), "annotation_recovery": annotation_details,
               "inherited_calling_contexts": len(seen_contexts), "inheritance_context_limit_reached": bool(pending),
               "mapped_instruction_categories": dict(Counter(value["category"] for value in exact.values())),
               "source_subsystems": dict(Counter(value["subsystem"] for value in exact.values())),
               "mapped_instruction_names": sum(value["category"] != "unknown" for value in exact.values())}
    return {"module": compiled["module"], "exact": exact, "aliases": dict(aliases), "details": details}


def _hlo_event(event, index):
    args = event.get("args", {})
    module = args.get("hlo_module") if isinstance(args, dict) else None
    if module is not None and not isinstance(module, str):
        return None, "invalid_hlo_module"
    if isinstance(module, str) and module and module != index["module"]:
        return None, "module_mismatch"
    op = args.get("hlo_op") if isinstance(args, dict) else None
    if op is not None and not isinstance(op, str):
        return None, "invalid_hlo_op"
    if isinstance(op, str):
        op = op.lstrip("%")
        if op in index["exact"]:
            return index["exact"][op], "hlo_op_exact"
        if not re.fullmatch(r"command_buffer(?:_\d+)?", op):
            return None, "hlo_op_unmatched"
        fallback = "command_buffer_kernel_symbol"
    else:
        fallback = "kernel_symbol_exact"
    name = str(event.get("name", "")).lstrip("%")
    values = index["aliases"].get(name, ())
    if values:
        return {"category": _consensus(value["category"] for value in values),
                "tags": sorted({tag for value in values for tag in value["tags"]}),
                "origins": sorted({origin for value in values for origin in value["origins"]}),
                "op_names": sorted({op_name for value in values for op_name in value["op_names"]})[:2],
                "subsystem": _consensus(value["subsystem"] for value in values),
                "subsystem_sources": sorted({sample for value in values for sample in value["subsystem_sources"]})[:2]}, fallback
    return None, "unmapped"


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _interval(event):
    start, duration = event.get("ts"), event.get("dur")
    if event.get("ph") != "X" or not _number(start) or not _number(duration) or duration <= 0:
        return None
    end = start + duration
    return (float(start), float(end)) if math.isfinite(end) else None


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _length(intervals):
    return sum(end - start for start, end in _merge(intervals))


def _clip(interval, windows):
    if windows is None:
        return [interval]
    start, end = interval
    return [(max(start, left), min(end, right)) for left, right in windows
            if min(end, right) > max(start, left)]


def _gpu_process(name):
    name = name.lower()
    if re.search(r"\b(host|cpu|executor|driver|api)\b", name):
        return False
    return bool(re.search(r"(?:/device:gpu:|\bgpu\s*[:#]?\s*\d+\b)", name))


def _stream(name):
    return (bool(re.search(r"\bstream\b", name, re.I)) and
            not bool(re.search(r"\b(cpu|host|executor|driver|api|all)\b", name, re.I)))


def _metadata(events):
    processes, threads = {}, {}
    for event in events:
        if event.get("ph") != "M":
            continue
        args = event.get("args", {})
        if not isinstance(args, dict) or event.get("pid") is None:
            continue
        name = args.get("name")
        if not isinstance(name, str):
            continue
        if event.get("name") == "process_name":
            processes[str(event.get("pid"))] = name
        elif event.get("name") == "thread_name" and event.get("tid") is not None:
            threads[(str(event.get("pid")), str(event.get("tid")))] = name
    streams = {key: {"pid": key[0], "tid": key[1], "process_name": processes.get(key[0], ""),
                     "thread_name": name}
               for key, name in threads.items() if _gpu_process(processes.get(key[0], "")) and _stream(name)}
    return processes, threads, streams


_COPY_DIRECTIONS = ("H2D", "D2H", "D2D", "unknown")
_CUDA_COPY_API = re.compile(r"^(?:cu|cuda)Memcpy[A-Za-z0-9_]*(?:\(.*\))?$", re.I)
_CUPTI_COPY_NAME = re.compile(r"^Memcpy(?:H(?:to|2)D|D(?:to|2)H|D(?:to|2)D)$", re.I)
_SYNC_API = re.compile(r"^(?:cuCtxSynchronize|cuStreamSynchronize|cuEventSynchronize|"
                       r"cudaDeviceSynchronize|cudaStreamSynchronize|cudaEventSynchronize|"
                       r"cuStreamWaitEvent|cudaStreamWaitEvent)(?:_v\d+)?(?:\(.*\))?$", re.I)


def _host_process(name):
    return not _gpu_process(name) and bool(re.search(r"\b(host|cpu)\b", name, re.I))


def _copy_fields(details):
    if isinstance(details, dict):
        return {str(key).lower(): value for key, value in details.items()}
    if isinstance(details, str):
        return {key.lower(): value for key, value in
                re.findall(r"\b([A-Za-z_]+)\s*[:=]\s*([^\s,;]+)", details)}
    return {}


def _copy_info(event):
    args = event.get("args", {})
    details = _copy_fields(args.get("memcpy_details"))
    if not details:
        details = {str(key).lower(): value for key, value in args.items()}
    kinds = {"host": "H", "pageable": "H", "pinned": "H", "pinned_host": "H", "device": "D"}
    source = details.get("kind_src", details.get("src_kind"))
    target = details.get("kind_dst", details.get("dst_kind"))
    pair = kinds.get(str(source).lower()), kinds.get(str(target).lower())
    field_direction = {("H", "D"): "H2D", ("D", "H"): "D2H", ("D", "D"): "D2D"}.get(pair)
    name = event.get("name", "")
    name_direction = None
    if _CUDA_COPY_API.fullmatch(name) or _CUPTI_COPY_NAME.fullmatch(name):
        for direction, expression in (("H2D", "HtoD|H2D|HostToDevice"),
                                      ("D2H", "DtoH|D2H|DeviceToHost"),
                                      ("D2D", "DtoD|D2D|DeviceToDevice")):
            if re.search(expression, name, re.I):
                name_direction = direction
                break
    if field_direction and name_direction and field_direction != name_direction:
        direction, direction_source = "unknown", "conflicting_details_and_name"
    else:
        direction = field_direction or name_direction or "unknown"
        direction_source = "explicit_memory_kinds" if field_direction else "explicit_copy_name" if name_direction else "unknown"
    byte_count, byte_source = None, "unknown"
    unit = str(details.get("unit", details.get("size_unit", "bytes"))).lower()
    if unit in ("b", "byte", "bytes"):
        for key in ("num_bytes", "bytes", "size_bytes", "size"):
            value = details.get(key)
            if ((isinstance(value, int) and not isinstance(value, bool) and value >= 0) or
                    isinstance(value, str) and re.fullmatch(r"\d+", value)):
                byte_count, byte_source = int(value), key
                break
    return direction, byte_count, {"direction_source": direction_source, "byte_source": byte_source,
                                   "memcpy_details": args.get("memcpy_details")}


def _complete_window_match(interval, windows):
    if windows is None or any(left <= interval[0] and interval[1] <= right for left, right in windows):
        return "complete"
    return "partial" if _clip(interval, windows) else "outside"


def _api_and_transfer_summary(events, processes, streams, windows, *, missing_window=False):
    """Observe copy/device and host API intervals without equating their costs."""
    device, host, sync = [], [], []
    exclusions = {name: Counter() for name in ("device", "host_api", "host_sync_api")}
    host_pids = {pid for pid, name in processes.items() if _host_process(name)}
    for event in events:
        args = event.get("args", {})
        if event.get("ph") != "X" or not isinstance(args, dict):
            continue
        key = str(event.get("pid")), str(event.get("tid"))
        name = str(event.get("name", ""))
        kind = None
        if key in streams and "memcpy_details" in args and "kernel_details" not in args:
            kind = "device"
        elif key[0] in host_pids:
            if _SYNC_API.fullmatch(name):
                kind = "host_sync_api"
            elif "kernel_details" not in args and (_CUDA_COPY_API.fullmatch(name) or
                  _CUPTI_COPY_NAME.fullmatch(name) and "memcpy_details" in args and
                  "correlation_id" in args and "kernel_details" not in args):
                kind = "host_api"
        if kind is None:
            continue
        interval = _interval(event)
        if interval is None:
            exclusions[kind]["invalid_events"] += 1
            continue
        match = "outside" if missing_window else _complete_window_match(interval, windows)
        if match != "complete":
            exclusions[kind][match + "_window_events_excluded"] += 1
            continue
        sample = {"name": name, "pid": key[0], "tid": key[1],
                  "start_us": interval[0], "end_us": interval[1],
                  "correlation_id": args.get("correlation_id")}
        if kind == "host_sync_api":
            sync.append((interval, sample))
        else:
            direction, byte_count, evidence = _copy_info(event)
            sample.update(evidence, bytes=byte_count)
            (device if kind == "device" else host).append((interval, direction, byte_count, sample))

    def times(intervals):
        return {"busy_seconds": _length(intervals) * 1e-6 if intervals else None,
                "event_duration_sum_seconds": sum(end-start for start, end in intervals) * 1e-6 if intervals else None}

    def copies(records, kind):
        intervals = [record[0] for record in records]
        known_bytes = sum(record[2] for record in records if record[2] is not None)
        unknown_bytes = sum(record[2] is None for record in records)
        value = {"supported": bool(records), "event_count": len(records),
                 "known_bytes": known_bytes, "bytes": known_bytes if records and not unknown_bytes else None,
                 "unknown_byte_events": unknown_bytes, **times(intervals), "by_direction": {},
                 "partial_window_events_excluded": exclusions[kind]["partial_window_events_excluded"],
                 "outside_window_events_excluded": exclusions[kind]["outside_window_events_excluded"],
                 "invalid_events": exclusions[kind]["invalid_events"]}
        for direction in _COPY_DIRECTIONS:
            selected = [record for record in records if record[1] == direction]
            amount = sum(record[2] for record in selected if record[2] is not None)
            unknown = sum(record[2] is None for record in selected)
            value["by_direction"][direction] = {
                "event_count": len(selected), "known_bytes": amount,
                "bytes": amount if selected and not unknown else None, "unknown_byte_events": unknown,
                **times([record[0] for record in selected]), "samples": [record[3] for record in selected[:3]]}
        if not records:
            value["reason"] = "No explicit complete copy events observed in the selected window; absence is not proof of zero transfer"
        return value

    sync_result = {"supported": bool(sync), "event_count": len(sync), **times([record[0] for record in sync]),
                   "by_name": {}, "samples": [record[1] for record in sync[:3]],
                   "partial_window_events_excluded": exclusions["host_sync_api"]["partial_window_events_excluded"],
                   "outside_window_events_excluded": exclusions["host_sync_api"]["outside_window_events_excluded"],
                   "invalid_events": exclusions["host_sync_api"]["invalid_events"],
                   "notes": ["These are explicit host synchronize/wait API intervals, not pure CPU blocking attribution.",
                             "A stream wait may enqueue a device dependency without blocking the host."]}
    for name in sorted({sample["name"] for _, sample in sync}):
        intervals = [interval for interval, sample in sync if sample["name"] == name]
        sync_result["by_name"][name] = {"event_count": len(intervals), **times(intervals)}
    if not sync:
        sync_result["reason"] = "No explicit complete host synchronize/wait API events observed; tracing support or blocking cost cannot be inferred"
    transfers = {"device": copies(device, "device"), "host_api": copies(host, "host_api"),
                 "window_policy": "complete_events_only" if windows is not None or missing_window else "whole_trace_no_host_window",
                 "notes": ["Device copies require memcpy_details and no kernel_details on an identified GPU stream.",
                           "Host API intervals and device intervals are separate and must not be added.",
                           "Direction unions can overlap; their sum is not aggregate wall time.",
                           "Known bytes are a lower bound when bytes is null; unknown directions retain their own bucket.",
                           "Events crossing window boundaries are excluded whole; bytes are never prorated."]}
    return transfers, sync_result


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_text(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return " ".join(map(_text, value))
    return ""


def _classify(event, prefix):
    # args.name carries the JAX named-scope path in current XProf exports.
    # hlo_op, tf_op and kernel names are also retained in the search text.
    text = str(event.get("name", "")) + " " + _text(event.get("args", {}))
    tags = sorted(set(re.findall(re.escape(prefix) + r"[A-Za-z0-9_.-]+", text)))
    if not tags:
        return "unknown", []
    names = [name[len(prefix):].lower().replace("-", "_") for name in tags]
    rollout = any(name.startswith(("rollout", "sampling")) for name in names)
    update = any(name.startswith(("update", "sgd", "minibatch", "gradient")) for name in names)
    leaf_aliases = {
        "physics": ("physics", "dynamics", "simulation", "mjx_step"),
        "reward": ("reward", "rewards"),
        "obs": ("obs", "observation", "observations"),
        "policy": ("policy", "actor", "actor_forward", "policy_forward"),
        "value": ("value", "critic", "critic_forward", "value_forward"),
    }
    leaves = {leaf for leaf, aliases in leaf_aliases.items()
              if any(name in aliases or any(name.endswith("." + alias) or name.endswith("_" + alias)
                                            for alias in aliases) for name in names)}
    explicit_other = {name for name in names if name in
                      ("batch_prepare", "normalizer", "optimizer", "shuffle", "bookkeeping")}
    # A kernel associated with two independent top-level phases is ambiguous.
    if rollout and update:
        return "mixed", tags
    if len(explicit_other) > 1:
        return "mixed", tags
    if explicit_other:
        selected = next(iter(explicit_other))
        if rollout or (selected in ("batch_prepare", "normalizer", "bookkeeping") and update):
            return "mixed", tags
        if selected in ("optimizer", "shuffle"):
            return "update." + selected, tags
        return selected, tags
    loss_leaves = {name for name in names if name in ("gae", "loss_terms", "actor_forward", "critic_forward")}
    if rollout or (leaves and not update and not loss_leaves):
        if len(leaves) > 1:
            return "mixed", tags
        return "rollout." + next(iter(leaves)) if leaves else "rollout.other", tags
    if any(name.startswith("gradient") for name in names):
        # Scope names inside value_and_grad do not by themselves prove that a
        # kernel is forward-only. Keep forward/backward work combined.
        return "update.gradient_combined", tags
    if update:
        if len(loss_leaves) > 1 or len(leaves) > 1:
            return "mixed", tags
        if loss_leaves:
            return "update." + next(iter(loss_leaves)), tags
        return "update.other", tags
    if len(loss_leaves) == 1:
        return "update." + next(iter(loss_leaves)), tags
    return "unknown", tags


def _partition(intervals):
    """Return additive category-only and cross-category-overlap durations."""
    edges = defaultdict(Counter)
    for category, pieces in intervals.items():
        for start, end in _merge(pieces):
            edges[start][category] += 1
            edges[end][category] -= 1
    active, exclusive, overlap = Counter(), defaultdict(float), defaultdict(float)
    previous = None
    for point in sorted(edges):
        if previous is not None and point > previous:
            categories = tuple(sorted(key for key, count in active.items() if count > 0))
            if len(categories) == 1:
                exclusive[categories[0]] += point - previous
            elif categories:
                overlap[" + ".join(categories)] += point - previous
        active.update(edges[point])
        previous = point
    return dict(exclusive), dict(overlap)


def summarize_trace(path, *, stage_prefix="ppo.", window_name=None, compiled_hlo_path=None,
                    annotated_mlir_path=None, source_stage_ranges=None):
    """Return measured device timing with explicit classification/overlap limits.

    ``window_name`` selects complete host annotations by exact name. With no
    explicit name, host events carrying ``step_num`` are selected. GPU work is
    clipped to their union. If no host annotation exists, GPU execution can
    still be summarized, but no host-wall/idle claim is made.
    """
    if not isinstance(stage_prefix, str) or not stage_prefix:
        raise ValueError("stage_prefix must be a nonempty string")
    if window_name is not None and (not isinstance(window_name, str) or not window_name):
        raise ValueError("window_name must be None or a nonempty string")
    if compiled_hlo_path is None and (annotated_mlir_path is not None or source_stage_ranges):
        raise ValueError("annotated_mlir_path/source_stage_ranges require compiled_hlo_path")
    hlo = _hlo_index(compiled_hlo_path, annotated_mlir_path, stage_prefix, source_stage_ranges) if compiled_hlo_path else None
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as stream:
        value = json.load(stream)
    events = value.get("traceEvents") if isinstance(value, dict) else value
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError("Expected a Chrome traceEvents list")
    processes, threads, streams = _metadata(events)
    windows, window_details = [], []
    for event in events:
        interval = _interval(event)
        pid = str(event.get("pid"))
        args = event.get("args", {})
        if interval is None or _gpu_process(processes.get(pid, "")) or not isinstance(args, dict):
            continue
        selected = event.get("name") == window_name if window_name is not None else "step_num" in args
        if selected:
            windows.append(interval)
            window_details.append({"name": str(event.get("name", "")), "pid": pid,
                                   "tid": str(event.get("tid")), "start_us": interval[0],
                                   "end_us": interval[1], "seconds": (interval[1] - interval[0]) * 1e-6,
                                   "step_num": args.get("step_num")})
    windows = _merge(windows)
    wall = _length(windows) * 1e-6 if windows else None
    transfers, host_sync_api = _api_and_transfer_summary(
        events, processes, streams, windows if windows else None,
        missing_window=window_name is not None and not windows)
    summary = {
        "schema_version": 1, "supported": False,
        "time_unit": "seconds", "trace_timestamp_unit": "microseconds",
        "display_time_unit": value.get("displayTimeUnit") if isinstance(value, dict) else None,
        "gpu_streams": list(streams.values()), "host_windows": window_details,
        "host_window_seconds": wall,
        "host_window_envelope_seconds": (windows[-1][1] - windows[0][0]) * 1e-6 if windows else None,
        "gpu_busy_seconds": None, "gpu_event_duration_sum_seconds": None,
        "category_busy_seconds": {}, "category_exclusive_seconds": {},
        "cross_category_overlap_seconds": None, "overlap_combinations_seconds": {},
        "unknown_or_mixed_busy_seconds": None, "classification_coverage_fraction": None,
        "gpu_idle_or_untraced_within_host_window_seconds": None,
        "gpu_event_count": 0, "invalid_device_event_count": 0, "top_kernels": [],
        "compiled_hlo_metadata": hlo["details"] if hlo else None,
        "classification_sources": {},
        "transfers": transfers, "host_sync_api": host_sync_api,
        "notes": ["Only complete events on metadata-identified GPU streams are measured.",
                  "Aggregate GPU busy is a union across streams/devices: at least one device is active; per-device busy is separate.",
                  "Category busy unions can overlap; exclusive categories plus cross-category overlap are additive.",
                  "Host wall minus traced GPU busy is not proof of CPU compute or a particular synchronization cause.",
                  "A gradient scope keeps forward/backward combined unless independently separable evidence exists."],
    }
    if window_name is not None and not windows:
        summary["reason"] = "Requested complete host window was not present: " + window_name
        return summary
    if not streams:
        summary["reason"] = "No metadata-identified GPU device streams; host/CUDA submission timing is insufficient"
        return summary
    intervals, kernel_intervals = defaultdict(list), defaultdict(list)
    kernel_counts, kernel_categories = Counter(), defaultdict(Counter)
    kernel_tags, kernel_hlo_names = defaultdict(set), defaultdict(list)
    kernel_sources, kernel_origins = defaultdict(Counter), defaultdict(Counter)
    subsystem_intervals, kernel_subsystems, kernel_subsystem_sources = defaultdict(list), defaultdict(Counter), defaultdict(set)
    classification_sources, hlo_cache = Counter(), {}
    device_intervals, event_duration_sum = defaultdict(list), 0.0
    all_intervals = []
    for event in events:
        key = str(event.get("pid")), str(event.get("tid"))
        if key not in streams or event.get("ph") != "X":
            continue
        interval = _interval(event)
        if interval is None:
            summary["invalid_device_event_count"] += 1
            continue
        pieces = _clip(interval, windows if windows else None)
        if not pieces:
            continue
        category, tags = _classify(event, stage_prefix)
        source = "trace_stage_tags" if tags else "unmapped"
        origins = []
        recovered_names = []
        subsystem = _source_subsystem(_text(event.get("args", {})))
        subsystem_sources = []
        if not tags and hlo:
            args = event.get("args", {})
            args = args if isinstance(args, dict) else {}
            key_hlo = repr(args.get("hlo_module")), repr(args.get("hlo_op")), event.get("name")
            if key_hlo not in hlo_cache:
                hlo_cache[key_hlo] = _hlo_event(event, hlo)
            recovered, source = hlo_cache[key_hlo]
            if recovered:
                category, tags = recovered["category"], recovered["tags"]
                origins, recovered_names = recovered["origins"], recovered["op_names"]
                if subsystem == "unknown":
                    subsystem, subsystem_sources = recovered["subsystem"], recovered["subsystem_sources"]
        classification_sources[source] += 1
        intervals[category].extend(pieces)
        subsystem_intervals[subsystem].extend(pieces)
        all_intervals.extend(pieces)
        device_intervals[key[0]].extend(pieces)
        name = str(event.get("name", "<unnamed>"))
        kernel_intervals[name].extend(pieces)
        kernel_counts[name] += 1
        kernel_categories[name][category] += 1
        kernel_sources[name][source] += 1
        kernel_origins[name].update(origins)
        kernel_subsystems[name][subsystem] += 1
        kernel_subsystem_sources[name].update(subsystem_sources)
        kernel_tags[name].update(tags)
        origin = event.get("args", {}).get("name") if isinstance(event.get("args"), dict) else None
        if isinstance(origin, str) and origin not in kernel_hlo_names[name] and len(kernel_hlo_names[name]) < 2:
            kernel_hlo_names[name].append(origin[:2048])
        for origin in recovered_names:
            if origin not in kernel_hlo_names[name] and len(kernel_hlo_names[name]) < 2:
                kernel_hlo_names[name].append(origin[:2048])
        summary["gpu_event_count"] += 1
        event_duration_sum += sum(end - start for start, end in pieces)
    if not all_intervals:
        summary["reason"] = "No valid complete GPU-stream execution events inside the selected windows"
        return summary
    busy = _length(all_intervals)
    exclusive, overlaps = _partition(intervals)
    unknown = _length([piece for category, pieces in intervals.items()
                       if category in _UNKNOWN or category.endswith(".mixed") for piece in pieces])
    domain_intervals = defaultdict(list)
    for category, pieces in intervals.items():
        domain_intervals[category.split(".", 1)[0]].extend(pieces)
    domain_exclusive, domain_overlaps = _partition(domain_intervals)
    domain_unknown = _length([piece for category in _UNKNOWN for piece in domain_intervals.get(category, ())])
    subsystem_exclusive, subsystem_overlaps = _partition(subsystem_intervals)
    subsystem_unknown = _length([piece for name, pieces in subsystem_intervals.items()
                                 if name in _UNKNOWN or name.endswith(".mixed") for piece in pieces])
    summary.update({
        "supported": True, "gpu_busy_seconds": busy * 1e-6,
        "gpu_event_duration_sum_seconds": event_duration_sum * 1e-6,
        "category_busy_seconds": {name: _length(pieces) * 1e-6 for name, pieces in sorted(intervals.items())},
        "category_exclusive_seconds": {name: seconds * 1e-6 for name, seconds in sorted(exclusive.items())},
        "cross_category_overlap_seconds": sum(overlaps.values()) * 1e-6,
        "overlap_combinations_seconds": {name: seconds * 1e-6 for name, seconds in sorted(overlaps.items())},
        "unknown_or_mixed_busy_seconds": unknown * 1e-6,
        "classification_coverage_fraction": max(0., 1. - unknown / busy),
        "domain_classification_coverage_fraction": max(0., 1. - domain_unknown / busy),
        "domain_busy_seconds": {name: _length(pieces) * 1e-6 for name, pieces in sorted(domain_intervals.items())},
        "domain_exclusive_seconds": {name: seconds * 1e-6 for name, seconds in sorted(domain_exclusive.items())},
        "domain_cross_overlap_seconds": sum(domain_overlaps.values()) * 1e-6,
        "classification_sources": dict(classification_sources),
        "subsystem_busy_seconds": {name: _length(pieces) * 1e-6 for name, pieces in sorted(subsystem_intervals.items())},
        "subsystem_exclusive_seconds": {name: seconds * 1e-6 for name, seconds in sorted(subsystem_exclusive.items())},
        "subsystem_cross_overlap_seconds": sum(subsystem_overlaps.values()) * 1e-6,
        "subsystem_coverage_fraction": max(0., 1. - subsystem_unknown / busy),
        "gpu_idle_or_untraced_within_host_window_seconds": max(0., wall - busy * 1e-6) if wall is not None else None,
        "gpu_busy_envelope_seconds": (max(end for _, end in all_intervals) -
                                       min(start for start, _ in all_intervals)) * 1e-6,
        "gpu_devices": [{"pid": pid, "name": processes[pid], "busy_seconds": _length(pieces) * 1e-6}
                        for pid, pieces in sorted(device_intervals.items())],
        "top_kernels": [{"name": name, "events": kernel_counts[name],
                         "busy_union_seconds": _length(pieces) * 1e-6,
                         "event_duration_sum_seconds": sum(end - start for start, end in pieces) * 1e-6,
                         "categories": dict(kernel_categories[name]), "stage_tags": sorted(kernel_tags[name]),
                         "classification_sources": dict(kernel_sources[name]),
                         "metadata_origins": dict(kernel_origins[name]), "hlo_name_samples": kernel_hlo_names[name],
                         "source_subsystems": dict(kernel_subsystems[name]),
                         "subsystem_source_samples": sorted(kernel_subsystem_sources[name])[:2]}
                        for name, pieces in sorted(kernel_intervals.items(),
                                                   key=lambda item: _length(item[1]), reverse=True)[:20]],
    })
    return summary


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--window-name")
    parser.add_argument("--compiled-hlo", type=Path)
    parser.add_argument("--annotated-mlir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = summarize_trace(args.trace, window_name=args.window_name,
                             compiled_hlo_path=args.compiled_hlo, annotated_mlir_path=args.annotated_mlir)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf8")
    print(text, end="")


if __name__ == "__main__":
    main()
