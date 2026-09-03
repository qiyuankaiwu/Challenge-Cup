"""演示服务接口的回归测试。"""

import base64
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

import config
import server
from orchestrator import Orchestrator, load_profile


class TestSessionPayload(unittest.TestCase):
    def setUp(self):
        self.key = "test-session"
        self.orch = Orchestrator()
        self.session = self.orch.run(load_profile("P-A"), max_kp=1)
        server.SESSIONS[self.key] = (self.orch, self.session)

    def tearDown(self):
        server.SESSIONS.pop(self.key, None)

    def test_resource_provenance_includes_verification_metadata(self):
        payload = server.session_payload(self.key)

        self.assertIn("kb", payload)
        source_id = payload["resources"][0]["claims"][0]["source_id"]
        source = payload["kb"][source_id]
        self.assertIn("source", source)
        self.assertIn("verified", source)
        self.assertIn("source_note", source)

    def test_quiz_answers_are_not_exposed_before_feedback(self):
        payload = server.session_payload(self.key)
        public_quiz = next(resource for resource in payload["resources"]
                           if resource["kind"] == "quiz")
        internal_quiz = next(resource for resource in self.session.resources
                             if resource.kind == "quiz")

        self.assertTrue(public_quiz["items"])
        self.assertNotIn("answer", public_quiz["items"][0])
        self.assertNotIn("explain", public_quiz["items"][0])
        self.assertIn("answer", internal_quiz.items[0])

    def test_feedback_choices_are_scored_against_internal_answers(self):
        quiz = next(resource for resource in self.session.resources
                    if resource.kind == "quiz")
        choices = [item["answer"] for item in quiz.items[:4]]

        correctness, results = server.score_feedback_choices(
            self.session, quiz.kp, choices)

        self.assertEqual([True] * len(choices), correctness)
        self.assertTrue(all(result["correct"] for result in results))
        self.assertTrue(all(type(result["answer"]) is bool for result in results))

    def test_feedback_rejects_truthy_strings(self):
        quiz = next(resource for resource in self.session.resources
                    if resource.kind == "quiz")
        with self.assertRaisesRegex(ValueError, "真正的布尔值"):
            server.score_feedback_choices(
                self.session, quiz.kp, ["false"] * len(quiz.items[:4]))


class TestApiResponseBoundary(unittest.TestCase):
    def setUp(self):
        self.existing_interviews = set(server.INTERVIEWS)
        self.existing_sessions = set(server.SESSIONS)
        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.thread = Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        for interview_id in set(server.INTERVIEWS) - self.existing_interviews:
            server.INTERVIEWS.pop(interview_id, None)
        for session_id in set(server.SESSIONS) - self.existing_sessions:
            server.SESSIONS.pop(session_id, None)

    def post(self, path: str, body: dict) -> dict:
        request = Request(
            f"http://127.0.0.1:{self.httpd.server_port}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_public_item_keeps_display_fields_and_hides_scoring_fields(self):
        internal = {
            "id": "Q-TEST",
            "kp": "KP-01",
            "level": 2,
            "stem": "测试题干",
            "options": ["甲", "乙", "丙", "丁"],
            "answer": 1,
            "explain": "内部解析",
            "source_id": "KB-001",
            "_reason": "select",
            "_kp_name": "机器人基础",
        }

        public = server.public_interview_item(internal)

        self.assertEqual("Q-TEST", public["id"])
        self.assertEqual(internal["options"], public["options"])
        self.assertEqual("select", public["_reason"])
        self.assertNotIn("answer", public)
        self.assertNotIn("explain", public)
        self.assertEqual(1, internal["answer"])

    def test_public_item_preserves_completed_marker(self):
        self.assertIsNone(server.public_interview_item(None))

    def test_start_and_answer_endpoints_never_return_the_next_answer(self):
        start = self.post("/api/interview/start", {
            "background": {"education": "本科", "hands_on_hours": 2},
            "max_items": 2,
        })
        first = start["item"]
        self.assertNotIn("answer", first)

        interview = server.INTERVIEWS[start["interview_id"]]
        correct_choice = interview.items[first["id"]]["answer"]
        follow_up = self.post("/api/interview/answer", {
            "interview_id": start["interview_id"],
            "item_id": first["id"],
            "choice": correct_choice,
        })

        self.assertTrue(follow_up["step"]["correct"])
        if follow_up["item"] is not None:
            self.assertNotIn("answer", follow_up["item"])

    def test_feedback_endpoint_scores_choices_then_reveals_results(self):
        initial = self.post("/api/run", {"profile_id": "P-A", "max_kp": 1})
        _, session = server.SESSIONS[initial["session_id"]]
        quiz = next(resource for resource in session.resources
                    if resource.kind == "quiz")
        choices = [item["answer"] for item in quiz.items[:4]]

        updated = self.post("/api/feedback", {
            "session_id": initial["session_id"],
            "kp": quiz.kp,
            "choices": choices,
        })

        self.assertTrue(all(item["correct"] for item in updated["feedback_result"]))
        public_quizzes = [resource for resource in updated["resources"]
                          if resource["kind"] == "quiz"]
        self.assertTrue(public_quizzes)
        self.assertTrue(all("answer" not in item
                            for resource in public_quizzes
                            for item in resource["items"]))


class TestMaterialStaging(unittest.TestCase):
    def test_staging_keeps_content_out_of_knowledge_base_and_unverified(self):
        text = (
            "工业机器人在示教器 T1 模式下调试时，应先确认急停按钮、防护围栏和安全"
            "距离处于可用状态。完成动作前需检查坐标系与工具参数，结束后记录报警现象"
            "及处理过程。本资料只用于受控暂存，必须由人工核对原始文件位置和外部依据，"
            "未经人工复核不得写入正式知识库，也不得标记为已核实。"
        )
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        before = config.KB_PATH.read_bytes()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = server.stage_material(
                "robot_t1.md", encoded, "团队已获授权的调试培训材料 V1.0", True,
                incoming_dir=root / "incoming", staging_root=root / "staged",
            )
            stage_dir = root / "staged" / report["upload_id"]
            manifest = json.loads((stage_dir / "upload_manifest.json").read_text(encoding="utf-8"))
            staged_lines = (stage_dir / "staged.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertFalse(report["knowledge_base_written"])
        self.assertFalse(report["verified"])
        self.assertEqual(manifest["review_state"], "pending_manual_review")
        self.assertFalse(manifest["knowledge_base_written"])
        self.assertFalse(manifest["verified"])
        self.assertTrue(staged_lines)
        self.assertTrue(all(json.loads(line)["verified"] is False for line in staged_lines))
        self.assertEqual(config.KB_PATH.read_bytes(), before)

    def test_staging_rejects_unsafe_or_unattested_input(self):
        encoded = base64.b64encode("足够长的测试资料内容。".encode("utf-8")).decode("ascii")
        with self.assertRaises(server.UploadError):
            server.stage_material("../unsafe.txt", encoded, "测试来源说明", True)
        with self.assertRaises(server.UploadError):
            server.stage_material("notes.exe", encoded, "测试来源说明", True)
        with self.assertRaises(server.UploadError):
            server.stage_material("notes.txt", encoded, "测试来源说明", False)
        with self.assertRaises(server.UploadError):
            server.stage_material("notes.txt", encoded, "无", True)

    def test_staging_rejects_a_file_when_nothing_can_be_extracted(self):
        encoded = base64.b64encode(b"%PDF-1.4\nnot a real PDF\n%%EOF").decode("ascii")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(server.UploadError, "未提取到可复核内容"):
                server.stage_material(
                    "broken.pdf", encoded, "待复核的测试 PDF", True,
                    incoming_dir=root / "incoming", staging_root=root / "staged",
                )

            self.assertEqual(1, len(list((root / "incoming").iterdir())))
            self.assertFalse((root / "staged").exists())


class TestFormalExaminerBoundary(unittest.TestCase):
    def test_examiner_only_uses_formal_demo_sources(self):
        examiner = server._examiner()
        if examiner is None:
            self.skipTest("命题审核器未启用")

        excluded = {"KB-015", "KB-016", "KB-018"}
        self.assertFalse(excluded & {chunk.id for chunk in examiner.retriever.chunks})


class _LandmarkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.elements = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])
            self.elements[values["id"]] = (tag, values)


class TestOnlineFrontend(unittest.TestCase):
    def test_learner_flow_landmarks_follow_the_task_order(self):
        parser = _LandmarkParser()
        parser.feed(Path("web/index.html").read_text(encoding="utf-8"))
        expected = [
            "workflowProgress", "learnerRecord", "learningPlan", "learningPath",
            "learningContent", "feedbackCard", "decisionPanel", "evidencePanel",
        ]
        self.assertEqual([item for item in parser.ids if item in expected], expected)

    def test_source_links_are_rendered_from_verified_provenance_text(self):
        path = Path("web/app.js")
        controller = path.read_text(encoding="utf-8") if path.exists() else ""

        self.assertIn("function sourceWithLink", controller)
        self.assertIn("sourceWithLink(kb.source)", controller)

    def test_async_alert_and_evidence_disclosure_have_accessible_defaults(self):
        parser = _LandmarkParser()
        parser.feed(Path("web/index.html").read_text(encoding="utf-8"))

        alert_tag, alert_attrs = parser.elements["uiAlert"]
        evidence_tag, evidence_attrs = parser.elements["evidencePanel"]
        self.assertEqual(alert_tag, "div")
        self.assertEqual(alert_attrs.get("role"), "alert")
        self.assertEqual(evidence_tag, "details")
        self.assertIn("hidden", evidence_attrs)
        self.assertNotIn("open", evidence_attrs)


if __name__ == "__main__":
    unittest.main()
