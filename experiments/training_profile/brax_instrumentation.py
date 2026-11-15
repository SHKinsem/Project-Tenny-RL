"""Add JAX name metadata to supported Brax stages without changing computation.

This module only rewrites in-memory Python function bodies. It never adds device
synchronization, callbacks or timers to a JIT, and never changes installed files.
The original AST must be recovered exactly by removing the inserted scopes.
"""
import ast
import copy
from contextlib import contextmanager
import functools
import hashlib
import inspect
import linecache
import textwrap
import types
import __future__


PREFIX = "ppo."


class UnsupportedBraxStructure(ValueError):
    """The installed source does not have the supported, unambiguous boundaries."""


def _path(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _path(node.value)
        return parent + "." + node.attr if parent else ""
    return ""


def _has_call(node, name):
    return any(isinstance(item, ast.Call) and _path(item.func) == name
               for item in ast.walk(node))


def _assignment_call(node, name):
    return (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and _path(node.value.func) == name)


def _targets(node):
    return {item.id for target in getattr(node, "targets", ())
            for item in ast.walk(target) if isinstance(item, ast.Name)}


def _single(items, predicate, label):
    matches = [index for index, value in enumerate(items) if predicate(value)]
    if len(matches) != 1:
        raise UnsupportedBraxStructure(f"Expected one {label}; found {len(matches)}")
    return matches[0]


def _function(body, name):
    index = _single(body, lambda node: isinstance(node, ast.FunctionDef) and node.name == name,
                    f"function {name}")
    value = body[index]
    if value.decorator_list:
        raise UnsupportedBraxStructure(f"Decorated function {name} is unsupported")
    return value


def _body(function):
    body = function.body
    has_doc = bool(body and isinstance(body[0], ast.Expr) and
                   isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str))
    return (body[:1], body[1:]) if has_doc else ([], body)


def _scope_name(node):
    if not isinstance(node, ast.With) or len(node.items) != 1:
        return None
    call = node.items[0].context_expr
    if (isinstance(call, ast.Call) and _path(call.func) == "jax.named_scope" and
            len(call.args) == 1 and not call.keywords and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str) and call.args[0].value.startswith(PREFIX)):
        return call.args[0].value
    return None


class _StripScopes(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        return node.body if _scope_name(node) else node


def strip_scopes(source):
    """Return an AST with only this module's reserved ppo.* scopes removed."""
    tree = ast.parse(textwrap.dedent(source)) if isinstance(source, str) else copy.deepcopy(source)
    return ast.fix_missing_locations(_StripScopes().visit(tree))


class _Annotator:
    def __init__(self):
        self.stages = []

    def wrap(self, body, stage, function):
        if not body:
            raise UnsupportedBraxStructure(f"Empty stage {stage} in {function}")
        label = PREFIX + stage
        self.stages.append({"name": label, "function": function,
                            "source_line": getattr(body[0], "lineno", None)})
        scope = ast.With(items=[ast.withitem(context_expr=ast.Call(
            func=ast.Attribute(value=ast.Name(id="jax", ctx=ast.Load()), attr="named_scope", ctx=ast.Load()),
            args=[ast.Constant(value=label)], keywords=[]))], body=body, type_comment=None)
        return ast.copy_location(scope, body[0])

    def train(self, tree):
        train = _function(tree.body, "train")
        step = _function(train.body, "training_step")
        shuffle = _function(train.body, "sgd_step")
        minibatch = _function(train.body, "minibatch_step")
        doc, body = _body(step)
        rollout = _single(body, lambda node: _assignment_call(node, "jax.lax.scan") and
                          {"state", "data"} <= _targets(node), "rollout scan")
        normalizer = _single(body, lambda node: isinstance(node, ast.Assign) and
                             _targets(node) == {"normalizer_params"}, "normalizer assignment")
        before_update = _single(body, lambda node: isinstance(node, ast.If) and
                                isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
                                and _path(node.test.operand) == "lr_is_adaptive_kl" and
                                _has_call(node, "running_statistics.update"), "pre-update normalizer branch")
        update = _single(body, lambda node: _assignment_call(node, "jax.lax.scan") and
                         {"optimizer_state", "params", "metrics"} <= _targets(node), "SGD update scan")
        after_update = _single(body, lambda node: isinstance(node, ast.If) and
                               _path(node.test) == "lr_is_adaptive_kl" and
                               _has_call(node, "running_statistics.update"), "post-update normalizer branch")
        if not (rollout + 1 < normalizer and before_update == normalizer + 1 and
                update == before_update + 1 and after_update == update + 1 and after_update + 1 < len(body)):
            raise UnsupportedBraxStructure("Unexpected training_step stage order")
        step.body = doc + [
            self.wrap(body[:rollout + 1], "rollout", "training_step"),
            self.wrap(body[rollout + 1:normalizer], "batch_prepare", "training_step"),
            self.wrap(body[normalizer:update], "normalizer", "training_step"),
            self.wrap(body[update:update + 1], "update", "training_step"),
            self.wrap(body[after_update:after_update + 1], "normalizer", "training_step"),
            self.wrap(body[after_update + 1:], "bookkeeping", "training_step"),
        ]
        doc, body = _body(shuffle)
        data = _single(body, lambda node: isinstance(node, ast.Assign) and
                       _targets(node) == {"shuffled_data"} and _has_call(node, "jax.tree_util.tree_map"),
                       "shuffled_data assignment")
        batch_scan = _single(body, lambda node: _assignment_call(node, "jax.lax.scan"), "minibatch scan")
        if batch_scan != data + 1:
            raise UnsupportedBraxStructure("Unexpected sgd_step shuffle/scan order")
        shuffle.body = doc + [self.wrap(body[:data + 1], "shuffle", "sgd_step")] + body[data + 1:]
        doc, body = _body(minibatch)
        gradient = _single(body, lambda node: _assignment_call(node, "loss_and_pgrad_fn"), "loss/gradient call")
        optimizer = _single(body, lambda node: _assignment_call(node, "optimizer.update"), "optimizer update")
        apply = _single(body, lambda node: _assignment_call(node, "optax.apply_updates"), "parameter update")
        if not (gradient < optimizer and apply == optimizer + 1):
            raise UnsupportedBraxStructure("Unexpected minibatch_step gradient/update order")
        minibatch.body = doc + body[:gradient] + [
            self.wrap(body[gradient:gradient + 1], "gradient", "minibatch_step"),
            self.wrap(body[gradient + 1:apply + 1], "optimizer", "minibatch_step"),
        ] + body[apply + 1:]

    def losses(self, tree):
        loss = _function(tree.body, "compute_ppo_loss")
        doc, body = _body(loss)
        actor = _single(body, lambda node: _assignment_call(node, "policy_apply"), "policy forward")
        critic = _single(body, lambda node: isinstance(node, ast.If) and
                         _path(node.test) == "use_distributional_critic" and _has_call(node, "value_apply"),
                         "critic forward branch")
        gae = _single(body, lambda node: _assignment_call(node, "compute_gae"), "GAE call")
        if not (actor + 1 == critic and critic + 1 < gae and gae + 1 < len(body)):
            raise UnsupportedBraxStructure("Unexpected compute_ppo_loss stage order")
        loss.body = doc + body[:actor] + [
            self.wrap(body[actor:actor + 1], "actor_forward", "compute_ppo_loss"),
            self.wrap(body[critic:critic + 1], "critic_forward", "compute_ppo_loss"),
            self.wrap(body[critic + 1:gae], "loss_terms", "compute_ppo_loss"),
            self.wrap(body[gae:gae + 1], "gae", "compute_ppo_loss"),
            self.wrap(body[gae + 1:], "loss_terms", "compute_ppo_loss"),
        ]


def annotate_source(source, module_kind):
    """Return (annotated_source, manifest) for supported train or loss source.

    Input may be one function or a complete source module. Only the target
    function is rewritten. The reserved ppo.* prefix rejects repeated annotation.
    """
    if module_kind not in ("train", "losses"):
        raise ValueError("module_kind must be 'train' or 'losses'")
    source = textwrap.dedent(source)
    original = ast.parse(source)
    if any(_scope_name(node) for node in ast.walk(original)):
        raise UnsupportedBraxStructure("Source already contains reserved ppo.* scope metadata")
    tree = copy.deepcopy(original)
    annotator = _Annotator()
    getattr(annotator, module_kind)(tree)
    ast.fix_missing_locations(tree)
    annotated = ast.unparse(tree) + "\n"
    stripped = strip_scopes(annotated)
    if ast.dump(stripped, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise AssertionError("Stage annotation changed the computation AST")
    return annotated, {"module_kind": module_kind,
                       "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                       "annotated_source_sha256": hashlib.sha256(annotated.encode()).hexdigest(),
                       "strip_scopes_ast_equal": True, "stages": annotator.stages}


def _compile_function(original, source, filename):
    if original.__closure__:
        raise UnsupportedBraxStructure("Only top-level module functions are supported")
    future_flags = 0
    for name in __future__.all_feature_names:
        future_flags |= getattr(__future__, name).compiler_flag
    module_code = compile(source, filename, "exec", flags=original.__code__.co_flags & future_flags,
                          dont_inherit=True)
    matches = [value for value in module_code.co_consts
               if isinstance(value, types.CodeType) and value.co_name == original.__name__]
    if len(matches) != 1:
        raise UnsupportedBraxStructure("Unable to identify annotated function code")
    # Do not exec the function definition: its default expressions may allocate
    # arrays or consume RNG. Reuse the original defaults and annotation objects.
    function = types.FunctionType(matches[0], original.__globals__, original.__name__, original.__defaults__)
    function.__kwdefaults__ = original.__kwdefaults__
    function.__annotations__ = original.__annotations__.copy()
    function.__dict__.update(original.__dict__)
    function.__module__ = original.__module__
    function.__qualname__ = original.__qualname__
    if inspect.signature(function) != inspect.signature(original):
        raise AssertionError("Stage annotation changed the public signature")
    return function


@contextmanager
def instrument_brax(train_module, losses_module):
    """Scope in-memory replacements; restore both functions even after failure."""
    targets = ((train_module, "train", "train"),
               (losses_module, "compute_ppo_loss", "losses"))
    prepared = []
    for module, name, kind in targets:
        original = getattr(module, name)
        if not inspect.isfunction(original):
            raise UnsupportedBraxStructure(f"Expected Python function {name}")
        source, manifest = annotate_source(inspect.getsource(original), kind)
        filename = f"<ppo_profile:{original.__module__}:{manifest['source_sha256'][:12]}>"
        function = _compile_function(original, source, filename)
        prepared.append((module, name, original, function, filename, source, manifest))
    cached = {}
    try:
        for module, name, original, function, filename, source, manifest in prepared:
            cached[filename] = linecache.cache.get(filename)
            linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
            setattr(module, name, function)
        yield {kind: prepared[index][-1] for index, (_, _, kind) in enumerate(targets)}
    finally:
        for module, name, original, _, filename, _, _ in reversed(prepared):
            setattr(module, name, original)
            previous = cached.get(filename)
            if previous is None:
                linecache.cache.pop(filename, None)
            else:
                linecache.cache[filename] = previous


def named_callable(function, name):
    """Add trace metadata to a callable; JAX is imported only when it is called."""
    if not isinstance(name, str) or not name.startswith(PREFIX) or len(name) == len(PREFIX):
        raise ValueError("Stage name must use a nonempty ppo.* label")
    @functools.wraps(function)
    def scoped(*args, **kwargs):
        import jax
        with jax.named_scope(name):
            return function(*args, **kwargs)
    return scoped
