"""Host-side diagnostics for the known (normalizer, policy, value) bundle.

Only the OUTER list/tuple wrapper is interchangeable. Network containers,
parameter paths, shapes and dtypes must still match. No weights go into JSON.
"""


def network_pair(params):
    if type(params) not in (list, tuple) or len(params) != 3:
        raise AssertionError("Expected (normalizer, policy, value) list/tuple bundle")
    return (params[1], params[2])


def _snapshot(params):
    import numpy as np

    nodes, leaves, values = [], [], {}

    def walk(value, path):
        name = type(value).__module__ + "." + type(value).__qualname__
        if type(value) is dict:
            if not all(isinstance(key, str) for key in value):
                raise AssertionError(f"Non-string network parameter key: {path}")
            nodes.append(dict(path=path, type=name, keys=sorted(value)))
            for key in sorted(value):
                walk(value[key], path + [key])
        elif type(value) in (list, tuple):
            nodes.append(dict(path=path, type=name, length=len(value)))
            for index, child in enumerate(value):
                walk(child, path + [index])
        else:
            # Arbitrary __array__ objects / custom pytree nodes could project
            # only some children. Accept actual array leaves, never projections.
            is_array = type(value) is np.ndarray
            if not is_array:
                try:
                    import jax
                except ImportError:
                    pass
                else:
                    is_array = isinstance(value, jax.Array)
            if not is_array:
                raise AssertionError(f"Unsupported network parameter node: {path} ({name})")
            array = np.asarray(value)
            # Network weights are real floating arrays, not strings/objects.
            if array.dtype.kind != "f":
                raise AssertionError(f"Non-floating network parameter: {path} ({array.dtype})")
            leaves.append(dict(path=path, shape=list(array.shape), dtype=str(array.dtype),
                               array_type=name, finite=bool(np.isfinite(array).all())))
            values[tuple(path)] = array

    for name, network in zip(("policy", "value"), network_pair(params), strict=True):
        walk(network, [name])
        if not any(path[0] == name for path in values):
            raise AssertionError(f"Empty {name} network parameter tree")
    return dict(bundle_type=type(params).__name__, nodes=nodes, leaves=leaves), values


def network_diagnostics(old, current):
    """Record actual container types, paths, leaf implementations/shapes/dtypes."""
    result = {}
    for name, params in (("before", old), ("after", current)):
        try:
            result[name] = _snapshot(params)[0]
        except (AssertionError, TypeError, ValueError) as exc:
            # Diagnostics must still be saved for malformed/unsupported input.
            # This is not acceptance: check_network_update fails on that input.
            result[name] = dict(bundle_type=type(params).__name__, diagnostic_error=str(exc))
    return result


def check_network_update(old, current):
    """Reject real schema/numerical changes while permitting further learning.

    JAX array and NumPy array leaf implementations may differ after loading;
    their dtype and shape may not. The caller also checks canonical JAX treedefs.
    Normalization is intentionally excluded: its changes do not prove learning.
    """
    import numpy as np

    left, before = _snapshot(old)
    right, after = _snapshot(current)
    if left["nodes"] != right["nodes"] or before.keys() != after.keys():
        raise AssertionError("Restore network parameter paths or containers changed")
    changed = 0
    max_change = 0.0
    for path, original in before.items():
        recovered = after[path]
        if original.shape != recovered.shape or original.dtype != recovered.dtype:
            raise AssertionError(f"Restore network shape/dtype changed: {list(path)}; "
                                 f"{original.shape}/{original.dtype} -> {recovered.shape}/{recovered.dtype}")
        if not np.isfinite(original).all() or not np.isfinite(recovered).all():
            raise AssertionError(f"Nonfinite restore network parameter: {list(path)}")
        if not np.array_equal(original, recovered):
            changed += 1
            max_change = max(max_change, float(np.max(np.abs(
                original.astype(np.float64) - recovered.astype(np.float64)))))
    if not np.isfinite(max_change):
        raise AssertionError("Nonfinite restore network parameter change")
    if not changed:
        raise AssertionError("Warm restart produced no changed policy/value network weights")
    return dict(warm_restart_network_leaves_checked=len(before),
                warm_restart_changed_network_leaves=changed,
                warm_restart_network_max_abs_change=max_change,
                warm_restart_outer_container_before=left["bundle_type"],
                warm_restart_outer_container_after=right["bundle_type"])
