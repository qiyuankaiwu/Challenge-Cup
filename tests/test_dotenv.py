"""项目根目录 `.env` 的模型配置加载行为。"""

from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from core.llm import MockLLM, RealLLM, build_llm
from core.llm import _load_project_env
from evalkit import doctor


class TestProjectDotenv(unittest.TestCase):

    def test_load_project_env_accepts_an_isolated_path(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "AGENTEDU_API_KEY=test-key\n"
                "AGENTEDU_BASE_URL=http://127.0.0.1:9999/v1\n"
                "AGENTEDU_MODEL=MiniMax-M3\n"
                "AGENTEDU_MODEL_STRONG=deepseek-v4-pro\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                _load_project_env(env_path)
                self.assertEqual(os.environ["AGENTEDU_API_KEY"], "test-key")
                self.assertEqual(os.environ["AGENTEDU_MODEL"], "MiniMax-M3")

    def test_build_llm_unquotes_project_dotenv_values(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                'AGENTEDU_API_KEY="test-key"\n'
                "AGENTEDU_BASE_URL='http://127.0.0.1:9999/v1'\n"
                'AGENTEDU_MODEL="test-model"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                _load_project_env(env_path)
                self.assertEqual(os.environ["AGENTEDU_API_KEY"], "test-key")
                self.assertEqual(
                    os.environ["AGENTEDU_BASE_URL"],
                    "http://127.0.0.1:9999/v1",
                )
                self.assertEqual(os.environ["AGENTEDU_MODEL"], "test-model")

    def test_build_llm_uses_exact_production_model_roles(self) -> None:
        with patch.dict(os.environ, {"AGENTEDU_API_KEY": "test-key"}, clear=True), \
                patch("core.llm._load_project_env"):
            llm = build_llm()
        self.assertIsInstance(llm, RealLLM)
        self.assertEqual(llm.model, "MiniMax-M3")
        self.assertEqual(llm.models, {"strong": "deepseek-v4-pro"})

    def test_build_llm_uses_distinct_provider_endpoints(self) -> None:
        environment = {
            "AGENTEDU_MINIMAX_API_KEY": "minimax-test-key",
            "AGENTEDU_MINIMAX_BASE_URL": "https://minimax.example/v1",
            "AGENTEDU_DEEPSEEK_API_KEY": "deepseek-test-key",
            "AGENTEDU_DEEPSEEK_BASE_URL": "https://deepseek.example",
        }
        with patch.dict(os.environ, environment, clear=True), \
                patch("core.llm._load_project_env"):
            llm = build_llm()

        self.assertIsInstance(llm, RealLLM)
        actual = {
            model_id: (adapter.base_url, adapter.api_key)
            for model_id, adapter in getattr(llm, "adapters", {}).items()
        }
        self.assertEqual(actual, {
            "MiniMax-M3": ("https://minimax.example/v1", "minimax-test-key"),
            "deepseek-v4-pro": ("https://deepseek.example", "deepseek-test-key"),
        })

    def test_build_llm_defaults_to_mainland_minimax_endpoint(self) -> None:
        environment = {
            "AGENTEDU_MINIMAX_API_KEY": "minimax-test-key",
            "AGENTEDU_DEEPSEEK_API_KEY": "deepseek-test-key",
        }
        with patch.dict(os.environ, environment, clear=True), \
                patch("core.llm._load_project_env"):
            llm = build_llm()

        self.assertEqual(
            llm.adapters["MiniMax-M3"].base_url,
            "https://api.minimaxi.com/v1",
        )
        self.assertEqual(
            llm.adapters["deepseek-v4-pro"].base_url,
            "https://api.deepseek.com",
        )

    def test_build_llm_reuses_adapter_for_legacy_unified_gateway(self) -> None:
        environment = {
            "AGENTEDU_API_KEY": "gateway-test-key",
            "AGENTEDU_BASE_URL": "https://gateway.example/v1",
        }
        with patch.dict(os.environ, environment, clear=True), \
                patch("core.llm._load_project_env"):
            llm = build_llm()

        self.assertIsInstance(llm, RealLLM)
        self.assertIs(
            llm.adapters["MiniMax-M3"],
            llm.adapters["deepseek-v4-pro"],
        )

    def test_build_llm_rejects_unknown_swapped_duplicate_and_blank_models(self) -> None:
        invalid = (
            ("unknown", "deepseek-v4-pro"),
            ("MiniMax-M3", "unknown"),
            ("deepseek-v4-pro", "MiniMax-M3"),
            ("MiniMax-M3", "MiniMax-M3"),
            ("", "deepseek-v4-pro"),
            ("MiniMax-M3", "   "),
        )
        for default_model, strong_model in invalid:
            with self.subTest(default=default_model, strong=strong_model), \
                    patch.dict(os.environ, {
                        "AGENTEDU_API_KEY": "test-key",
                        "AGENTEDU_MODEL": default_model,
                        "AGENTEDU_MODEL_STRONG": strong_model,
                    }, clear=True), patch("core.llm._load_project_env"):
                with self.assertRaisesRegex(ValueError, "生产模型配置"):
                    build_llm()

    def test_doctor_points_offline_users_to_project_dotenv(self) -> None:
        output = io.StringIO()

        with patch("evalkit.doctor.build_llm", return_value=MockLLM()), \
                redirect_stdout(output):
            doctor.main()

        message = output.getvalue()
        self.assertIn("仓库根目录的 .env", message)
        self.assertIn("AGENTEDU_MINIMAX_API_KEY", message)
        self.assertIn("AGENTEDU_DEEPSEEK_API_KEY", message)
        self.assertNotIn("export AGENTEDU_API_KEY", message)


if __name__ == "__main__":
    unittest.main()
