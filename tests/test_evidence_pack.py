"""Tests for integrity, execution and safety boundaries of evidence packages."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools.evidence_pack import (
    CheckResult,
    CheckSpec,
    EvidenceError,
    EvidenceRunner,
    PackageManifest,
    atomic_write_json,
    copy_package,
    hash_inputs,
    read_manifest,
    sha256_file,
    summarize,
    validate_package,
)


class TestCheckSpec(unittest.TestCase):
    def test_rejects_empty_identifier(self):
        with self.assertRaises(ValueError):
            CheckSpec("", ("echo", "x"), "log.txt", "x")

    def test_rejects_identifier_with_spaces(self):
        with self.assertRaises(ValueError):
            CheckSpec("bad id", ("echo", "x"), "log.txt", "x")

    def test_rejects_empty_command(self):
        with self.assertRaises(ValueError):
            CheckSpec("test", (), "log.txt", "x")

    def test_rejects_absolute_log(self):
        with self.assertRaises(ValueError):
            CheckSpec("test", ("echo",), str(Path.cwd()), "x")

    def test_rejects_parent_log(self):
        with self.assertRaises(ValueError):
            CheckSpec("test", ("echo",), "../log.txt", "x")

    def test_rejects_zero_timeout(self):
        with self.assertRaises(ValueError):
            CheckSpec("test", ("echo",), "log.txt", "x", timeout_seconds=0)


class EvidencePackTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "root"
        self.package = Path(self.temp.name) / "package"
        self.root.mkdir()
        self.package.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def make_manifest(self, checks=None):
        return PackageManifest(
            created_at="2026-01-01T00:00:00+00:00", root=".", source_revision="abc",
            environment={"PYTHONUTF8": "1"}, inputs={}, checks=checks or [],
            limitations=["automated checks are not human verification"],
        )

    def write_manifest(self, manifest):
        atomic_write_json(self.package / "manifest.json", manifest.as_dict())


class TestHashes(EvidencePackTestCase):
    def test_sha256_is_stable(self):
        path = self.root / "input.txt"
        path.write_text("evidence", encoding="utf-8")
        self.assertEqual(sha256_file(path), sha256_file(path))

    def test_hash_inputs_uses_relative_posix_paths(self):
        nested = self.root / "nested"
        nested.mkdir()
        path = nested / "input.txt"
        path.write_text("evidence", encoding="utf-8")
        self.assertEqual(hash_inputs(self.root, [path]), {"nested/input.txt": sha256_file(path)})

    def test_hash_inputs_rejects_missing_file(self):
        with self.assertRaises(EvidenceError):
            hash_inputs(self.root, [self.root / "missing.txt"])

    def test_hash_inputs_rejects_external_path(self):
        external = Path(self.temp.name) / "external.txt"
        external.write_text("x", encoding="utf-8")
        with self.assertRaises(EvidenceError):
            hash_inputs(self.root, [external])

    def test_atomic_json_is_valid_and_terminated(self):
        path = self.package / "nested" / "value.json"
        atomic_write_json(path, {"中文": [1, 2]})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"中文": [1, 2]})
        self.assertTrue(path.read_bytes().endswith(b"\n"))


class TestRunner(EvidencePackTestCase):
    def spec(self, identifier="ok", outputs=()):
        return CheckSpec(identifier, ("{python}", "-c", "print('ok')"),
                         f"logs/{identifier}.log", "prints ok", outputs)

    def test_records_successful_command(self):
        runner = EvidenceRunner(self.root, self.package)
        result = runner.run(self.spec())
        self.assertTrue(result.passed)
        self.assertEqual(result.return_code, 0)
        self.assertIn("print", (self.package / result.log).read_text(encoding="utf-8"))

    def test_records_failed_command(self):
        spec = CheckSpec("bad", ("{python}", "-c", "raise SystemExit(3)"),
                         "logs/bad.log", "fails")
        result = EvidenceRunner(self.root, self.package).run(spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.return_code, 3)

    def test_resume_does_not_reexecute_existing_log(self):
        spec = self.spec()
        log = self.package / spec.log
        log.parent.mkdir(parents=True)
        log.write_text("old", encoding="utf-8")
        result = EvidenceRunner(self.root, self.package).run(spec, resume=True)
        self.assertEqual(result.status, "resumed")
        self.assertEqual(log.read_text(encoding="utf-8"), "old")

    def test_fail_fast_raises_after_failure(self):
        bad = CheckSpec("bad", ("{python}", "-c", "raise SystemExit(2)"),
                        "logs/bad.log", "fails")
        with self.assertRaises(EvidenceError):
            EvidenceRunner(self.root, self.package).run_all([bad])

    def test_non_fail_fast_retains_later_results(self):
        bad = CheckSpec("bad", ("{python}", "-c", "raise SystemExit(2)"),
                        "logs/bad.log", "fails")
        results = EvidenceRunner(self.root, self.package).run_all(
            [bad, self.spec()], fail_fast=False)
        self.assertEqual([result.status for result in results], ["failed", "passed"])

    def test_duplicate_check_ids_are_rejected(self):
        with self.assertRaises(EvidenceError):
            EvidenceRunner(self.root, self.package).run_all([self.spec(), self.spec()])

    def test_output_hash_is_recorded_after_command(self):
        target = self.package / "out.txt"
        code = f"from pathlib import Path; Path({str(target)!r}).write_text('done')"
        spec = CheckSpec("out", (sys.executable, "-c", code), "logs/out.log", "writes", ("out.txt",))
        result = EvidenceRunner(self.root, self.package).run(spec)
        self.assertEqual(result.outputs, {"out.txt": sha256_file(target)})


class TestManifest(EvidencePackTestCase):
    def test_round_trips_manifest(self):
        manifest = self.make_manifest([CheckResult("a", ["x"], "a.log", "s", "f", 1, 0, "passed")])
        self.write_manifest(manifest)
        self.assertEqual(read_manifest(self.package / "manifest.json").source_revision, "abc")

    def test_rejects_missing_manifest_field(self):
        atomic_write_json(self.package / "manifest.json", {"created_at": "x"})
        with self.assertRaises(EvidenceError):
            read_manifest(self.package / "manifest.json")

    def test_validation_requires_manifest(self):
        self.assertEqual(validate_package(self.package), ["missing manifest.json"])

    def test_validation_requires_log(self):
        self.write_manifest(self.make_manifest([
            CheckResult("a", ["x"], "missing.log", "s", "f", 1, 0, "passed")]))
        self.assertEqual(validate_package(self.package), ["missing log for a: missing.log"])

    def test_validation_detects_invalid_status(self):
        (self.package / "a.log").write_text("x", encoding="utf-8")
        self.write_manifest(self.make_manifest([
            CheckResult("a", ["x"], "a.log", "s", "f", 1, 0, "unknown")]))
        self.assertIn("invalid status for a: unknown", validate_package(self.package))

    def test_validation_detects_hash_mismatch(self):
        output = self.package / "result.json"
        output.write_text("first", encoding="utf-8")
        (self.package / "a.log").write_text("x", encoding="utf-8")
        result = CheckResult("a", ["x"], "a.log", "s", "f", 1, 0, "passed",
                             {"result.json": sha256_file(output)})
        self.write_manifest(self.make_manifest([result]))
        output.write_text("second", encoding="utf-8")
        self.assertIn("output hash mismatch: result.json", validate_package(self.package))

    def test_summary_counts_statuses(self):
        manifest = self.make_manifest([
            CheckResult("a", ["x"], "a", "s", "f", 1.25, 0, "passed"),
            CheckResult("b", ["x"], "b", "s", "f", 2.25, 1, "failed"),
        ])
        report = summarize(manifest)
        self.assertEqual(report["statuses"], {"passed": 1, "failed": 1, "resumed": 0})
        self.assertEqual(report["duration_seconds"], 3.5)

    def test_copy_requires_valid_source(self):
        destination = Path(self.temp.name) / "copy"
        with self.assertRaises(EvidenceError):
            copy_package(self.package, destination)
