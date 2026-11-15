"""Pure-Python checks for profiling metadata and scoped restoration."""
import ast
from contextlib import contextmanager
import inspect
import linecache
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brax_instrumentation import (UnsupportedBraxStructure, annotate_source,
                                  instrument_brax, named_callable, strip_scopes)


TRAIN = '''
def train(value=3, *, fail=False, return_nested=False):
    """Training public documentation."""
    def minibatch_step(carry, data, normalizer_params):
        """Minibatch documentation."""
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), grads = loss_and_pgrad_fn(params, normalizer_params, data, key_loss)
        if lr_is_adaptive_kl:
            lr = ppo_optimizer.adaptive_kl_learning_rate(optimizer_state)
        else:
            lr = learning_rate
        metrics['learning_rate'] = lr
        params_update, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, params_update)
        return (optimizer_state, params, key), metrics
    def sgd_step(carry, unused_t, data, normalizer_params):
        """Shuffle documentation."""
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)
        def convert_data(x):
            return jax.random.permutation(key_perm, x)
        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            minibatch_step, (optimizer_state, params, key_grad), shuffled_data)
        return (optimizer_state, params, key), metrics
    def training_step(carry, unused_t):
        """Step documentation."""
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)
        policy = make_policy(training_state.params)
        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(env, current_state, policy, current_key, unroll_length)
            return (next_state, next_key), data
        (state, _), data = jax.lax.scan(f, (state, key_generate_unroll), ())
        data = jax.tree_util.tree_map(lambda x: x, data)
        assert data.discount.shape[1:] == (unroll_length,)
        if bootstrap_on_timeout:
            data = data.replace(reward=data.reward + 1)
        normalizer_params = training_state.normalizer_params
        if not lr_is_adaptive_kl:
            normalizer_params = running_statistics.update(normalizer_params, data.observation)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            sgd_step, (training_state.optimizer_state, training_state.params, key_sgd), ())
        if lr_is_adaptive_kl:
            normalizer_params = running_statistics.update(normalizer_params, data.observation)
        new_training_state = TrainingState(optimizer_state, params, normalizer_params)
        return (new_training_state, state, new_key), metrics
    if fail:
        raise RuntimeError('expected failure')
    if return_nested:
        return training_step, minibatch_step, sgd_step
    return value
'''

LOSSES = '''
def compute_ppo_loss(value: int = 4, *, skip=True):
    """Loss public documentation."""
    if skip:
        return value
    policy_apply = network.policy
    value_apply = network.value
    data = convert(data)
    policy_logits = policy_apply(params, data)
    if use_distributional_critic:
        baseline, quantiles = value_apply(params, data)
        bootstrap_value, _ = value_apply(params, data[-1])
    else:
        baseline = value_apply(params, data)
        bootstrap_value = value_apply(params, data[-1])
        quantiles = None
    rewards = data.reward * reward_scaling
    truncation = data.truncation
    vs, advantages = compute_gae(rewards=rewards, values=baseline)
    total_loss = sum(advantages) + sum(vs)
    return total_loss, {'total_loss': total_loss}
'''


def computation(source):
    return ast.dump(ast.parse(source), include_attributes=False)


class InstrumentationTests(unittest.TestCase):
    def module(self, name, source, **extra):
        module = types.ModuleType(name)
        module.__dict__.update(extra)
        filename = f"<test_{name}>"
        previous = linecache.cache.get(filename)
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        def cleanup():
            if previous is None:
                linecache.cache.pop(filename, None)
            else:
                linecache.cache[filename] = previous
        self.addCleanup(cleanup)
        exec(compile(source, filename, "exec"), module.__dict__)
        return module

    def test_train_ast_is_unchanged_after_removing_only_added_scopes(self):
        marked, manifest = annotate_source(TRAIN, "train")
        self.assertEqual(ast.dump(strip_scopes(marked), include_attributes=False), computation(TRAIN))
        self.assertTrue(manifest["strip_scopes_ast_equal"])
        self.assertEqual(len(manifest["source_sha256"]), 64)
        self.assertEqual({item["name"] for item in manifest["stages"]}, {
            "ppo.rollout", "ppo.batch_prepare", "ppo.normalizer", "ppo.update",
            "ppo.bookkeeping", "ppo.shuffle", "ppo.gradient", "ppo.optimizer"})

    def test_loss_ast_unchanged_and_all_forward_gae_terms_marked(self):
        marked, manifest = annotate_source(LOSSES, "losses")
        self.assertEqual(ast.dump(strip_scopes(marked), include_attributes=False), computation(LOSSES))
        self.assertEqual({item["name"] for item in manifest["stages"]}, {
            "ppo.actor_forward", "ppo.critic_forward", "ppo.gae", "ppo.loss_terms"})
        self.assertEqual(sum(item["name"] == "ppo.loss_terms" for item in manifest["stages"]), 2)

    def test_added_calls_are_named_scope_only(self):
        for source, kind in ((TRAIN, "train"), (LOSSES, "losses")):
            marked, manifest = annotate_source(source, kind)
            old_calls = sum(isinstance(node, ast.Call) for node in ast.walk(ast.parse(source)))
            new_calls = sum(isinstance(node, ast.Call) for node in ast.walk(ast.parse(marked)))
            self.assertEqual(new_calls - old_calls, len(manifest["stages"]))
            self.assertNotIn("block_until_ready", marked)
            self.assertNotIn("debug.callback", marked)

    def test_full_module_preserves_unrelated_functions_imports_and_globals(self):
        original = "import math\nCONST = 8\ndef unrelated():\n    return CONST\n" + TRAIN
        marked, _ = annotate_source(original, "train")
        self.assertEqual(ast.dump(strip_scopes(marked), include_attributes=False), computation(original))

    def test_repeated_annotation_rejected(self):
        marked, _ = annotate_source(TRAIN, "train")
        with self.assertRaisesRegex(UnsupportedBraxStructure, "already contains"):
            annotate_source(marked, "train")

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            annotate_source(TRAIN, "other")

    def test_missing_or_ambiguous_boundaries_rejected(self):
        mutations = (
            TRAIN.replace("def training_step(", "def renamed_step("),
            TRAIN.replace("normalizer_params = training_state.normalizer_params", "other = training_state.normalizer_params"),
            TRAIN.replace("        normalizer_params = training_state.normalizer_params\n",
                          "        normalizer_params = training_state.normalizer_params\n" * 2),
            TRAIN.replace("optimizer.update(grads, optimizer_state)", "another.update(grads, optimizer_state)"),
            TRAIN.replace("loss_and_pgrad_fn(params, normalizer_params, data, key_loss)", "another_fn(params, normalizer_params, data, key_loss)"),
            TRAIN.replace("shuffled_data = jax.tree_util.tree_map", "renamed_data = jax.tree_util.tree_map"),
            TRAIN.replace("if not lr_is_adaptive_kl:", "if other_condition:"),
        )
        for value in mutations:
            with self.subTest(source=value[:50]), self.assertRaises(UnsupportedBraxStructure):
                annotate_source(value, "train")
        for value in (LOSSES.replace("policy_apply(params, data)", "unknown(params, data)"),
                      LOSSES.replace("compute_gae(rewards=", "unknown_gae(rewards=")):
            with self.assertRaises(UnsupportedBraxStructure):
                annotate_source(value, "losses")

    def test_stage_order_drift_rejected(self):
        tree = ast.parse(TRAIN)
        step = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == "training_step")
        indices = [i for i, n in enumerate(step.body) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                   and n.test.id == "lr_is_adaptive_kl"]
        index = indices[0]
        step.body[index], step.body[index - 1] = step.body[index - 1], step.body[index]
        with self.assertRaisesRegex(UnsupportedBraxStructure, "order"):
            annotate_source(ast.unparse(tree), "train")

    def test_context_restores_identity_after_success(self):
        training = self.module("fake_train_success", TRAIN)
        losses = self.module("fake_losses_success", LOSSES)
        old_train, old_loss = training.train, losses.compute_ppo_loss
        with instrument_brax(training, losses) as manifest:
            self.assertIsNot(training.train, old_train)
            self.assertIsNot(losses.compute_ppo_loss, old_loss)
            self.assertEqual(training.train(), 3)
            self.assertEqual(losses.compute_ppo_loss(), 4)
            self.assertEqual(inspect.signature(training.train), inspect.signature(old_train))
            self.assertEqual(training.train.__doc__, old_train.__doc__)
            self.assertEqual(losses.compute_ppo_loss.__annotations__, old_loss.__annotations__)
            self.assertEqual(set(manifest), {"train", "losses"})
            for nested in training.train(return_nested=True):
                self.assertTrue(nested.__doc__)
        self.assertIs(training.train, old_train)
        self.assertIs(losses.compute_ppo_loss, old_loss)

    def test_context_restores_after_body_exception(self):
        training = self.module("fake_train_error", TRAIN)
        losses = self.module("fake_losses_error", LOSSES)
        old_train, old_loss = training.train, losses.compute_ppo_loss
        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            with instrument_brax(training, losses):
                training.train(fail=True)
        self.assertIs(training.train, old_train)
        self.assertIs(losses.compute_ppo_loss, old_loss)

    def test_setup_failure_makes_no_partial_patch(self):
        training = self.module("fake_train_setup", TRAIN)
        losses = self.module("fake_losses_setup", LOSSES.replace("compute_gae(rewards=", "changed_gae(rewards="))
        old_train, old_loss = training.train, losses.compute_ppo_loss
        with self.assertRaises(UnsupportedBraxStructure):
            with instrument_brax(training, losses):
                self.fail("invalid setup entered")
        self.assertIs(training.train, old_train)
        self.assertIs(losses.compute_ppo_loss, old_loss)

    def test_default_values_are_not_evaluated_again(self):
        calls = []
        sentinel = object()
        def default():
            calls.append(True)
            return sentinel
        training = self.module("fake_defaults", TRAIN.replace("value=3", "value=default()"), default=default)
        losses = self.module("fake_loss_defaults", LOSSES)
        original_defaults = training.train.__defaults__
        self.assertEqual(len(calls), 1)
        with instrument_brax(training, losses):
            self.assertEqual(len(calls), 1)
            self.assertIs(training.train.__defaults__, original_defaults)
            self.assertIs(training.train(), sentinel)

    def test_future_annotations_preserved_without_reevaluation(self):
        training = self.module("fake_future_train", "from __future__ import annotations\n" + TRAIN)
        losses = self.module("fake_future_loss", "from __future__ import annotations\n" + LOSSES)
        old_annotations = losses.compute_ppo_loss.__annotations__.copy()
        self.assertEqual(old_annotations["value"], "int")
        with instrument_brax(training, losses):
            self.assertEqual(losses.compute_ppo_loss.__annotations__, old_annotations)

    def test_nested_context_is_rejected_and_outer_remains_installed(self):
        training = self.module("fake_nested_train", TRAIN)
        losses = self.module("fake_nested_loss", LOSSES)
        original = training.train
        with instrument_brax(training, losses):
            outer = training.train
            with self.assertRaises(UnsupportedBraxStructure):
                with instrument_brax(training, losses):
                    self.fail("nested setup entered")
            self.assertIs(training.train, outer)
        self.assertIs(training.train, original)

    def test_named_callable_preserves_signature_result_and_exception_cleanup(self):
        events = []
        @contextmanager
        def scope(name):
            events.append(("enter", name))
            try:
                yield
            finally:
                events.append(("exit", name))
        def function(value, *, fail=False):
            """Callable documentation."""
            if fail:
                raise RuntimeError("failure")
            return value + 1
        wrapped = named_callable(function, "ppo.physics")
        self.assertEqual(inspect.signature(wrapped), inspect.signature(function))
        self.assertEqual(wrapped.__doc__, function.__doc__)
        with patch.dict(sys.modules, {"jax": types.SimpleNamespace(named_scope=scope)}):
            self.assertEqual(wrapped(8), 9)
            with self.assertRaises(RuntimeError):
                wrapped(8, fail=True)
        self.assertEqual(events, [("enter", "ppo.physics"), ("exit", "ppo.physics")] * 2)
        for name in ("", "physics", "ppo.", None):
            with self.assertRaises(ValueError):
                named_callable(function, name)

    def test_actual_local_source_snapshot_if_available(self):
        directory = Path(__file__).resolve().parents[3] / "tasks/results/ppo-stage-profiling/production"
        if not (directory / "brax-train.py").exists():
            self.skipTest("local installed-source snapshot is not distributed")
        for name, kind in (("brax-train.py", "train"), ("brax-losses.py", "losses")):
            source = (directory / name).read_text(encoding="utf8")
            marked, manifest = annotate_source(source, kind)
            self.assertTrue(manifest["strip_scopes_ast_equal"])
            self.assertEqual(ast.dump(strip_scopes(marked), include_attributes=False), computation(source))

    def test_installed_brax_context_without_running_training_if_available(self):
        try:
            from brax.training.agents.ppo import train as training, losses
        except ModuleNotFoundError:
            self.skipTest("Brax is not installed in this interpreter")
        original_train, original_loss = training.train, losses.compute_ppo_loss
        with instrument_brax(training, losses) as manifest:
            self.assertEqual(inspect.signature(training.train), inspect.signature(original_train))
            self.assertEqual(inspect.signature(losses.compute_ppo_loss), inspect.signature(original_loss))
            self.assertIs(training.train.__globals__, original_train.__globals__)
            self.assertIs(training.train.__defaults__, original_train.__defaults__)
            self.assertTrue(manifest["train"]["strip_scopes_ast_equal"])
            self.assertTrue(manifest["losses"]["strip_scopes_ast_equal"])
        self.assertIs(training.train, original_train)
        self.assertIs(losses.compute_ppo_loss, original_loss)


if __name__ == "__main__":
    unittest.main()
