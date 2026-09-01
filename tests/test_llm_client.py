"""模型客户端的健壮性。

接真模型时最常见的三类故障 —— 限流、JSON 模式不支持、模型名写错 ——
在全流程里表现出来都是"生成结果为空"，极难定位。所以在客户端这一层
就要处理掉，并且要有测试盯着，不能等到答辩现场才发现。

用本地假端点测，不联网。
"""

import config
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core.llm import TASK_TIER, LLMError, RealLLM

STATE = {"hits": 0, "mode": "normal"}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        STATE["hits"] += 1
        STATE.setdefault("last", {})
        STATE["last"] = body
        mode = STATE["mode"]
        if mode == "no_json" and "response_format" in body:
            msg = json.dumps({"error": {"message": "response_format is not supported"}}).encode()
            self.send_response(400)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        if mode == "flaky" and STATE["hits"] % 3 != 0:
            self.send_response(429)
            self.end_headers()
            return
        if mode == "auth":
            self.send_response(401)
            self.end_headers()
            return
        out = json.dumps({
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class TestRealLLMResilience(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 8393), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        time.sleep(0.2)
        cls.url = "http://127.0.0.1:8393"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        STATE["hits"] = 0
        STATE["mode"] = "normal"

    def _llm(self, **kw):
        # 这组测试量的是网络行为（重试、降级、计费），必须关掉缓存，
        # 否则第二次相同请求会命中缓存、根本不发出去，测到的就不是网络了。
        kw.setdefault("cache", False)
        return RealLLM(self.url, "k", "default-model", timeout=5, **kw)

    def test_normal_call(self):
        self.assertIn("ok", self._llm().run("verify", "s", "u"))

    def test_task_routing_picks_tier_model(self):
        llm = self._llm(models={"strong": "big", "light": "small"})
        self.assertEqual(llm.model_for("make_item"), "big")       # strong
        self.assertEqual(llm.model_for("synthesize"), "small")    # light

    def test_missing_tier_falls_back_to_default(self):
        llm = self._llm(models={"strong": "big"})
        self.assertEqual(llm.model_for("draft_claims"), "default-model")

    def test_item_generation_uses_strongest_tier(self):
        """命题出错代价最高，必须走强模型。"""
        self.assertEqual(TASK_TIER["make_item"], "strong")

    def test_json_mode_downgrades_once_and_remembers(self):
        """端点不支持 response_format 时降级，且不再反复白试。"""
        STATE["mode"] = "no_json"
        llm = self._llm()
        self.assertIn("ok", llm.run("verify", "只输出JSON", "u", json_mode=True))
        self.assertFalse(llm._json_mode_ok)
        before = STATE["hits"]
        llm.run("verify", "只输出JSON", "u", json_mode=True)
        self.assertEqual(STATE["hits"] - before, 1, "降级后不应再试 response_format")

    def test_downgrade_puts_constraint_in_prompt(self):
        STATE["mode"] = "no_json"
        llm = self._llm()
        llm.run("verify", "系统提示", "u", json_mode=True)
        self.assertIn("只输出 JSON", STATE["last"]["messages"][0]["content"])

    def test_retries_on_rate_limit(self):
        STATE["mode"] = "flaky"
        llm = self._llm(retries=4)
        self.assertIn("ok", llm.run("simplify", "s", "u"))
        self.assertGreaterEqual(STATE["hits"], 2)

    def test_auth_error_is_not_retried(self):
        """401 重试多少次都没用，快速失败比慢慢磨好。"""
        STATE["mode"] = "auth"
        llm = self._llm(retries=3)
        with self.assertRaises(LLMError):
            llm.run("simplify", "s", "u")
        self.assertEqual(STATE["hits"], 1)

    def test_usage_and_cost_accounted(self):
        llm = self._llm(price_in=1.0, price_out=2.0)
        llm.run("verify", "s", "u")
        st = llm.stats()
        self.assertEqual(st["tokens_in"], 100)
        self.assertEqual(st["tokens_out"], 25)
        self.assertAlmostEqual(st["cost_cny"], 100 / 1e6 * 1 + 25 / 1e6 * 2, places=6)
        self.assertEqual(st["by_task"]["verify"]["calls"], 1)

    def test_failure_is_counted(self):
        STATE["mode"] = "auth"
        llm = self._llm(retries=1)
        with self.assertRaises(LLMError):
            llm.run("simplify", "s", "u")
        self.assertEqual(llm.stats()["failures"], 1)


class TestBuildLLM(unittest.TestCase):

    def test_no_key_falls_back_to_mock(self):
        import os
        from core.llm import MockLLM, build_llm
        old = os.environ.pop("AGENTEDU_API_KEY", None)
        try:
            self.assertIsInstance(build_llm(), MockLLM)
        finally:
            if old:
                os.environ["AGENTEDU_API_KEY"] = old


class TestFreeTierGuards(unittest.TestCase):
    """免费额度下的三条保护：限速、缓存、调用上限。

    免费档的真实约束不是"能不能用"，是每分钟请求数和总额度。
    这三条没有的话，一次演示很容易把额度打光，而且失败方式很难看。
    """

    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 8394), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        time.sleep(0.2)
        cls.url = "http://127.0.0.1:8394"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        STATE["hits"] = 0
        STATE["mode"] = "normal"

    def test_cache_avoids_duplicate_requests(self):
        llm = RealLLM(self.url, "k", "m", timeout=5, cache=True)
        for _ in range(5):
            llm.run("draft_claims", "固定系统提示", "同一输入")
        self.assertEqual(STATE["hits"], 1, "相同请求应只发一次")
        self.assertEqual(llm.stats()["cache_hits"], 4)

    def test_cache_distinguishes_different_input(self):
        llm = RealLLM(self.url, "k", "m", timeout=5, cache=True)
        for i in range(3):
            llm.run("draft_claims", "固定系统提示", f"输入{i}")
        self.assertEqual(STATE["hits"], 3)

    def test_cache_can_be_disabled(self):
        llm = RealLLM(self.url, "k", "m", timeout=5, cache=False)
        for _ in range(3):
            llm.run("verify", "s", "u")
        self.assertEqual(STATE["hits"], 3)

    def test_throttle_spaces_out_requests(self):
        llm = RealLLM(self.url, "k", "m", timeout=5, rpm=120, cache=False)
        t0 = time.monotonic()
        for i in range(3):
            llm.run("verify", "s", f"u{i}")
        # rpm=120 → 间隔 0.5s，三次至少 1.0s
        self.assertGreaterEqual(time.monotonic() - t0, 0.9)

    def test_budget_stops_calling_and_flags(self):
        llm = RealLLM(self.url, "k", "m", timeout=5, budget_calls=2, cache=False)
        llm.run("verify", "s", "a")
        llm.run("verify", "s", "b")
        with self.assertRaises(LLMError):
            llm.run("verify", "s", "c")
        self.assertTrue(llm.stats()["budget_hit"])
        self.assertEqual(STATE["hits"], 2, "超限后不应再发请求")

    def test_item_generation_degrades_instead_of_crashing(self):
        """额度耗尽时命题必须返回 None 回退题库，而不是把链路炸掉。

        命题是增强项，没有它系统照样能用固定题库跑完。
        演示现场额度用完还能继续演，和当场白屏，是两回事。
        """
        import json as _json
        from agents.examiner import ExaminerAgent
        from core.retrieval import Retriever
        llm = RealLLM(self.url, "k", "m", timeout=5, budget_calls=0, cache=False)
        llm.budget_calls = 1
        R = Retriever.from_jsonl(config.KB_PATH)
        kps = _json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]
        ex = ExaminerAgent(llm, R, {k["id"]: k for k in kps})
        results = [ex.make_item(k["id"], 3) for k in kps[:3]]
        self.assertTrue(any(r is None for r in results))
        self.assertGreaterEqual(ex.llm_errors, 1)

    def test_intake_still_works_without_model(self):
        """自述解析在模型不可用时必须仍有结果 —— 规则通路本来就在。"""
        from agents.intake import IntakeAgent
        llm = RealLLM(self.url, "k", "m", timeout=5, budget_calls=1, cache=False)
        ia = IntakeAgent(llm)
        for _ in range(3):
            bg = ia.parse("机械专业大三，实操大概40小时")
            self.assertEqual(bg["education"], "本科")
            self.assertEqual(bg["hands_on_hours"], 40)
