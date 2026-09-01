"""离线演示页的回归护栏。"""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "showcase.template.html"
SHOWCASE = ROOT / "web" / "showcase.html"


class TestOfflineShowcase(unittest.TestCase):
    """保护离线入口的状态机与自包含约束。"""

    def test_answer_feedback_does_not_preload_next_item(self) -> None:
        """动态追问必须由用户点击“下一题”后才消费。"""
        source = TEMPLATE.read_text(encoding="utf-8")
        start = source.index("function answerQuestion(choice){")
        end = source.index("function finishQuiz(){", start)
        answer_function = source[start:end]

        self.assertIn("const finished = IV.done()", answer_function)
        self.assertNotIn("IV.next()", answer_function)

    def test_generated_showcase_contains_embedded_runtime(self) -> None:
        """最终离线文件应内联快照和浏览器端推理引擎。"""
        source = SHOWCASE.read_text(encoding="utf-8")

        self.assertIn('<script id="snapshot" type="application/json">', source)
        self.assertIn("global.Engine = {", source)
        self.assertNotIn("__SNAPSHOT__", source)
        self.assertNotIn("__ENGINE__", source)

    def test_resources_expose_full_content_and_provenance_status(self) -> None:
        """三类资源不能只显示标题或断言摘要。"""
        source = TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("完整资源正文", source)
        self.assertIn("resourceBody(r)", source)
        self.assertIn("r.body", source)
        self.assertIn("r.items", source)
        self.assertIn("人工已核实", source)
        self.assertIn("待人工核实", source)


if __name__ == "__main__":
    unittest.main()
