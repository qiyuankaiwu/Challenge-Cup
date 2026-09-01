"""Regression tests for the G2 evidence-freezing utility."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import freeze_g2


class TestFreezeG2(unittest.TestCase):
    def test_refuses_to_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                freeze_g2.freeze(output)

    def test_writes_manifest_and_honest_limitations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "evidence"
            commands: list[tuple[list[str], Path]] = []

            def record(command: list[str], log_path: Path) -> None:
                commands.append((command, log_path))

            with patch.object(freeze_g2, "_run", side_effect=record), \
                    patch.object(freeze_g2, "_output", return_value="available"):
                freeze_g2.freeze(output)

            self.assertEqual(len(commands), 8)
            self.assertTrue(all(path.is_relative_to(output) for _, path in commands))
            manifest = json.loads(
                (output / "基线与复现信息.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["evaluation"], {"cases": 50, "seed": 20260905})
            self.assertEqual(len(manifest["limitations"]), 3)
            self.assertIn("不构成独立领域真值", manifest["limitations"][0])

    def test_resume_skips_existing_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "evidence"
            log_path = output / "测试通过记录" / "单元测试.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text("already complete", encoding="utf-8")
            commands: list[list[str]] = []

            def record(command: list[str], _: Path) -> None:
                commands.append(command)

            with patch.object(freeze_g2, "_run", side_effect=record), \
                    patch.object(freeze_g2, "_output", return_value="available"):
                freeze_g2.freeze(output, resume=True)

            self.assertEqual(len(commands), 7)
            self.assertNotIn([freeze_g2.sys.executable, "-m", "unittest", "discover",
                              "-s", "tests", "-v"], commands)
