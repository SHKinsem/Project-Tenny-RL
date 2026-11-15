"""Real temporary file copies exercise storage and model publication boundaries."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
storage_module = importlib.import_module("workbench_storage")
Storage = storage_module.WorkbenchStorage
StorageError = storage_module.StorageError


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workbench-storage-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts"
        self.storage = Storage(self.artifacts)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def source(self, name="original_run", mode="original"):
        run = self.root / "external" / name
        checkpoint = run / "checkpoints" / "000000000128"
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights.bin").write_bytes(b"real-file-copy-fixture")
        (run / "policy.cpp.txt").write_text("float weights[] = {1.0f};", encoding="utf-8")
        for filename in ("policy_metadata.json", "env_config.json", "ppo_config.json"):
            self.write(run / filename, {"mode": mode})
        self.write(run / "report.json", {"mode": mode, "checkpoint": str(checkpoint),
                                         "export_path": str(run / "policy.cpp.txt"), "steps": 128})
        self.write(run / "status.json", {"phase": "completed", "run_dir": str(run), "pid": 123,
                                         "checkpoint": str(checkpoint)})
        self.write(run / "workbench-run.json", {"runId": run.name, "id": "keep-context-id"})
        self.write(checkpoint.with_name(checkpoint.name + ".workbench.json"),
                   {"createdByRunId": run.name, "id": "keep-checkpoint-id", "trainedSeconds": 600})
        return run, checkpoint

    def validator(self, run, mode):
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["mode"], mode)
        self.assertEqual(Path(report["checkpoint"]).parent.parent, run)
        self.assertEqual(Path(report["export_path"]).parent, run)
        self.assertEqual((Path(report["checkpoint"]) / "weights.bin").read_bytes(), b"real-file-copy-fixture")
        self.assertTrue(Path(report["export_path"]).is_file())
        return {"checkpoint": report["checkpoint"], "verified": True}

    def test_default_reads_do_not_create_or_move_storage(self):
        self.assertEqual(self.storage.save_root(), self.artifacts / "console")
        self.assertFalse(self.artifacts.exists())
        self.assertFalse(self.storage.describe()["configured"])

    def test_description_uses_only_runtime_distribution_identity(self):
        with patch.dict("os.environ", {"WSL_DISTRO_NAME": "Ubuntu-production"}):
            self.assertEqual(self.storage.describe()["wslDistro"], "Ubuntu-production")

    def test_missing_or_invalid_import_path_is_a_storage_error(self):
        for candidate in (None, {}, [], "", "relative"):
            with self.subTest(path=candidate), self.assertRaises(StorageError):
                self.storage.import_checkpoint(candidate, "original", self.validator)

    def test_external_save_location_registration_preserves_previous_locations(self):
        first, second = self.root / "first", self.root / "second"
        first.mkdir()
        second.mkdir()
        self.storage.configure(first)
        self.storage.configure(second)
        refreshed = Storage(self.artifacts)
        self.assertEqual(refreshed.save_root(), second)
        self.assertIn(first, refreshed.registered_roots())
        self.assertIn(self.artifacts / "console", refreshed.registered_roots())
        self.assertIn(self.artifacts / "workbench-imports", refreshed.registered_roots())

    def test_invalid_save_paths_are_rejected_without_changing_configuration(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        self.storage.configure(allowed)
        original = self.storage.config_path.read_bytes()
        for candidate in ("relative", str(self.root / "missing"), self.root.anchor, self.root / ".." / "escape"):
            with self.subTest(candidate=candidate), self.assertRaises(StorageError):
                self.storage.configure(candidate)
            self.assertEqual(self.storage.config_path.read_bytes(), original)

    def test_missing_external_save_location_is_not_silently_recreated(self):
        external = self.root / "removable-storage"
        external.mkdir()
        self.storage.configure(external)
        external.rmdir()
        with self.assertRaises(StorageError):
            self.storage.save_root()
        self.assertFalse(external.exists())

    def test_registered_model_directories_are_exact_and_bounded(self):
        root = self.artifacts / "console"
        run = root / "original" / "run_001"
        checkpoint = run / "checkpoints" / "000000000128"
        checkpoint.mkdir(parents=True)
        self.assertEqual(self.storage.resolve_run(run, "original"), run)
        for candidate in (root, root / "original", run, run / "checkpoints", checkpoint):
            self.assertEqual(self.storage.resolve_folder(candidate), candidate)
        private = run / "other"
        private.mkdir()
        with self.assertRaises(StorageError):
            self.storage.resolve_folder(private)
        with self.assertRaises(StorageError):
            self.storage.resolve_run(run, "torque200")

    def test_import_preserves_identity_metadata_and_source_bytes(self):
        source, checkpoint = self.source()
        report_before = (source / "report.json").read_bytes()
        result = self.storage.import_checkpoint(checkpoint, "original", self.validator)
        destination = Path(result["runDir"])
        self.assertEqual(destination.name, source.name)
        self.assertTrue(result["ok"])
        self.assertEqual((source / "report.json").read_bytes(), report_before)
        sidecar = Path(result["checkpoint"] + ".workbench.json")
        self.assertEqual(json.loads(sidecar.read_text())["id"], "keep-checkpoint-id")
        self.assertEqual(json.loads((destination / "workbench-run.json").read_text())["runId"], source.name)
        status = json.loads((destination / "status.json").read_text())
        self.assertNotIn("pid", status)
        self.assertEqual(status["run_dir"], str(destination))
        self.assertEqual(self.storage.resolve_run(destination, "original"), destination)

    def test_import_validation_dto_and_records_use_published_paths(self):
        source, _ = self.source()
        calls = []
        def validator(run, mode):
            self.validator(run, mode)
            calls.append(run)
            report = json.loads((run / "report.json").read_text())
            return {"run_dir": str(run), "checkpoint": report["checkpoint"],
                    "report_path": report["report_path"], "export_path": report["export_path"]}
        result = self.storage.import_checkpoint(source, "original", validator)
        final = Path(result["runDir"])
        self.assertNotEqual(calls[0], final)
        self.assertEqual(calls[-1], final)
        self.assertEqual(result["model"]["run_dir"], str(final))
        self.assertEqual(result["model"]["report_path"], str(final / "report.json"))
        self.assertTrue(Path(result["model"]["report_path"]).is_file())
        self.assertNotIn(".import_", json.dumps(result))

    def test_import_rejects_duplicate_or_nonfinite_metadata_before_publication(self):
        source, _ = self.source()
        for raw in ('{"mode":"original","nested":{"value":NaN}}', '{"value":1e999}',
                    '{"mode":"original","mode":"torque200"}', '{"nested":{"key":1,"key":2}}'):
            with self.subTest(metadata=raw):
                (source / "runtime_profile.json").write_text(raw, encoding="utf-8")
                with self.assertRaises(StorageError):
                    self.storage.import_checkpoint(source, "original", self.validator)
                parent = self.storage.import_root / "original"
                self.assertFalse(parent.exists() and list(parent.iterdir()))

    def test_invalid_export_field_is_a_storage_error(self):
        source, _ = self.source()
        report = json.loads((source / "report.json").read_text())
        for value in (None, {}, []):
            report["export_path"] = value
            self.write(source / "report.json", report)
            with self.subTest(export=value), self.assertRaises(StorageError):
                self.storage.import_checkpoint(source, "original", self.validator)

    def test_import_rebases_an_exported_run_moved_from_another_machine(self):
        source, checkpoint = self.source()
        report = json.loads((source / "report.json").read_text())
        report["checkpoint"] = "/old-machine/run/checkpoints/" + checkpoint.name
        report["export_path"] = "/old-machine/run/policy.cpp.txt"
        self.write(source / "report.json", report)
        result = self.storage.import_checkpoint(source, "original", self.validator)
        self.assertTrue(Path(result["checkpoint"]).is_dir())

    def test_repeat_same_import_is_idempotent_and_changed_content_cannot_replace_it(self):
        source, checkpoint = self.source()
        first = self.storage.import_checkpoint(source, "original", self.validator)
        again = self.storage.import_checkpoint(source, "original", self.validator)
        self.assertTrue(again["alreadyImported"])
        self.assertEqual(again["checkpoint"], first["checkpoint"])
        self.write(source / "env_config.json", {"mode": "original", "changed": True})
        with self.assertRaises(StorageError):
            self.storage.import_checkpoint(source, "original", self.validator)
        self.assertTrue(Path(first["checkpoint"]).is_dir())

    def test_incompatible_mode_or_validator_never_publishes_a_model(self):
        source, _ = self.source()
        with self.assertRaises(StorageError):
            self.storage.import_checkpoint(source, "torque200", self.validator)
        with self.assertRaises(StorageError):
            self.storage.import_checkpoint(source, "original", lambda *_: False)
        mode_root = self.storage.import_root / "original"
        self.assertEqual(list(mode_root.iterdir()), [])

    def test_final_validation_failure_removes_only_new_import(self):
        source, _ = self.source()
        calls = []
        def validate(run, mode):
            calls.append(run)
            return len(calls) == 1
        with self.assertRaises(StorageError):
            self.storage.import_checkpoint(source, "original", validate)
        self.assertTrue(source.is_dir())
        self.assertEqual(list((self.storage.import_root / "original").iterdir()), [])

    def test_incomplete_or_unverified_checkpoint_is_rejected(self):
        source, checkpoint = self.source()
        (source / "ppo_config.json").unlink()
        with self.assertRaises(StorageError):
            self.storage.import_checkpoint(source, "original", self.validator)
        self.write(source / "ppo_config.json", {})
        different = checkpoint.with_name("000000000064")
        different.mkdir()
        with self.assertRaises(StorageError):
            self.storage.import_checkpoint(different, "original", self.validator)

    def test_copy_limits_abort_without_publication(self):
        source, _ = self.source()
        with patch.object(storage_module, "MAX_IMPORT_BYTES", 2), self.assertRaises(StorageError):
            self.storage.import_checkpoint(source, "original", self.validator)
        self.assertEqual(list((self.storage.import_root / "original").iterdir()), [])

    def test_symbolic_links_are_rejected_in_storage_or_model(self):
        source, checkpoint = self.source()
        link = checkpoint / "linked.bin"
        try:
            link.symlink_to(checkpoint / "weights.bin")
        except OSError:
            self.skipTest("symlink creation requires platform permission")
        with self.assertRaises(StorageError):
            self.storage.import_checkpoint(source, "original", self.validator)
        directory_link = self.root / "linked-directory"
        directory_link.symlink_to(source, target_is_directory=True)
        with self.assertRaises(StorageError):
            self.storage.configure(directory_link)


if __name__ == "__main__":
    unittest.main()
