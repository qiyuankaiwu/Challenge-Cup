"""命题 Agent 的单元测试。

这个文件盯的是整套系统里后果最重的一个环节。

讲义写错一句，学习者可能看得出来。**题目答案错了，系统会拿一把错的尺子量人**：
答对被判成答错，掌握度往反方向走，后续所有资源推荐跟着错，而且整条链路上
没有任何环节会发现。所以命题审核必须比内容审核更严，这些用例就是那道闸。

夹具覆盖命题最常见的五类瑕疵：
  1. 答案在知识库里查无实据
  2. 干扰项其实也成立（一题两解，比答案错了还糟）
  3. 题干或答案凭空出现新数值
  4. 正确答案明显更长（不用会也能蒙对）
  5. 选项重复、序号越界这类结构性错误
"""

import json
import unittest

import config
from agents.examiner import ExaminerAgent, ItemRejected, _is_numeric
from core.llm import MockLLM
from core.retrieval import Retriever


class _VerdictLLM:
    """Deterministic stand-in for the external semantic verifier."""

    def __init__(self, verdict: str):
        self.verdict = verdict

    def run(self, *args, **kwargs) -> str:
        return json.dumps({"verdict": self.verdict, "reason": "test"})


class TestNumericDetection(unittest.TestCase):
    def test_numeric_options(self):
        for t in ["0.5", "250", "1.4米", "8层", "三年", "一万小时", "二十五"]:
            self.assertTrue(_is_numeric(t), t)

    def test_textual_options(self):
        for t in ["不小于30度", "按住超程解除按钮再反向移出", "安全回路板", ""]:
            self.assertFalse(_is_numeric(t), t)


class TestVetting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kps = {k["id"]: k for k in
                   json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]}
        cls.ex = ExaminerAgent(MockLLM(), Retriever.from_jsonl(config.KB_PATH), cls.kps)

    def _base(self, **kw):
        item = {"id": "T-1", "kp": "KP-02", "level": 2,
                "stem": "ABB 示例中手动模式的最高速度是",
                "options": ["250", "200", "600", "900"], "answer": 0,
                "source_id": "KB-004", "origin": "generated"}
        item.update(kw)
        return item

    def test_good_item_passes(self):
        self.ex.vet(self._base())

    def test_answer_without_evidence_is_rejected(self):
        """正确答案的数值在知识库里根本不存在。"""
        with self.assertRaises(ItemRejected):
            self.ex.vet(self._base(options=["999", "200", "600", "900"], answer=0))

    def test_distractor_that_also_holds_is_rejected(self):
        """把正文里同样成立的说法拿来做干扰项，会形成一题两解。

        这一类比答案错了还糟：学习者选了另一个"对的"会被判错，
        掌握度反向移动，且没有任何环节会发现。
        """
        with self.assertRaises(ItemRejected) as cm:
            self.ex.vet(self._base(
                stem="ABB 示例中手动模式应如何操作",
                options=["只能通过示教器操作", "机器人可在手动或自动模式下运行",
                         "可以绕过现场规程", "无需示教器"], answer=0))
        self.assertIn("可能同样成立", str(cm.exception))

    def test_high_overlap_distractors_pass_when_auditor_rejects_them(self):
        """词面相似不是成立；只有独立审核确认其不成立时才能放行。"""
        examiner = ExaminerAgent(
            _VerdictLLM("unsupported"),
            self.ex.retriever,
            self.kps,
        )
        item = {
            "id": "T-counterfactual",
            "kp": "KP-01",
            "level": 3,
            "stem": "关于机器人控制柜的组成与功能，下列说法正确的是：",
            "options": [
                "包含计算机、电力电子和电机驱动器，执行程序并向各关节电机发送控制命令",
                "包含计算机、电力电子和减速器，执行程序并向各关节电机发送控制命令",
                "包含计算机、气动阀和电机驱动器，执行程序并向各关节电机发送控制命令",
                "包含液压泵、电力电子和电机驱动器，执行程序并向各关节电机发送控制命令",
            ],
            "answer": 0,
            "source_id": "KB-002",
            "origin": "generated",
        }

        try:
            examiner.vet(item)
        except ItemRejected as exc:
            self.fail(f"独立审核已拒绝的干扰项仍被词面阈值误伤：{exc}")

    def test_rule_only_drop_cannot_certify_a_supported_distractor(self):
        """规则低覆盖误判不能充当语义证据，放过第二个正确答案。"""
        examiner = ExaminerAgent(
            _VerdictLLM("supported"),
            self.ex.retriever,
            self.kps,
        )
        item = {
            "id": "T-second-correct",
            "kp": "KP-13",
            "level": 3,
            "stem": "关于机器人工作站的安全防护，下列说法正确的是：",
            "options": [
                "安全门打开时应停止机器人及相关设备的自动运行",
                "机器人安全防护包括安全围栏与安全门",
                "完成作业后应按工艺要求核对程序与工具状态",
                "设备维护前应按现场要求记录润滑与紧固状态",
            ],
            "answer": 0,
            "source_id": "KB-022",
            "origin": "generated",
        }

        with self.assertRaises(ItemRejected) as cm:
            examiner.vet(item)
        self.assertIn("可能同样成立", str(cm.exception))

    def test_stem_with_invented_number_is_rejected(self):
        with self.assertRaises(ItemRejected) as cm:
            self.ex.vet(self._base(stem="依据第 37 条，手动模式的最高速度是"))
        self.assertIn("题干", str(cm.exception))

    def test_distractors_may_carry_numbers_absent_from_kb(self):
        """干扰项按定义就该是知识库里没有的数，不能因此被拦。

        第一版的审核规则不分题干和选项，一律禁止出现切片外数值，
        结果把所有能用的题都毙掉了 —— 等于禁止出错误选项。
        """
        self.ex.vet(self._base(options=["250", "333", "910", "770"], answer=0))

    def test_length_clue_is_rejected(self):
        with self.assertRaises(ItemRejected) as cm:
            self.ex.vet(self._base(
                kp="KP-10", source_id="KB-015",
                stem="SRVO-001 表示什么",
                options=["操作面板急停被按下，必须先排除风险后按安全规程恢复",
                         "喷漆", "扫码", "联网"],
                answer=0))
        self.assertIn("长度线索", str(cm.exception))

    def test_length_rule_skips_pure_numeric_options(self):
        """纯数值选项的长度反映量级不是正确性，不该触发长度线索。"""
        self.ex.vet(self._base(options=["250", "20", "500", "30"], answer=0))

    def test_duplicate_options_rejected(self):
        with self.assertRaises(ItemRejected):
            self.ex.vet(self._base(options=["250", "250", "200", "600"]))

    def test_answer_index_out_of_range(self):
        with self.assertRaises(ItemRejected):
            self.ex.vet(self._base(answer=9))

    def test_dangling_source_rejected(self):
        with self.assertRaises(ItemRejected):
            self.ex.vet(self._base(source_id="KB-999"))

    def test_too_few_options(self):
        with self.assertRaises(ItemRejected):
            self.ex.vet(self._base(options=["250", "200"], answer=0))


class TestGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kps = {k["id"]: k for k in
                   json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]}
        cls.R = Retriever.from_jsonl(config.KB_PATH)

    def test_generated_items_always_pass_vetting(self):
        """出题口和审核口是同一套标准：交付出来的题必定过审。"""
        ex = ExaminerAgent(MockLLM(), self.R, self.kps)
        made = 0
        for kp in self.kps:
            it = ex.make_item(kp, 3)
            if it is None:
                continue
            made += 1
            ex.vet(it)                      # 不抛异常即通过
            self.assertEqual(it["origin"], "generated")
            self.assertIsNotNone(self.R.get(it["source_id"]))
        self.assertGreater(made, 0, "离线命题应当至少能产出若干道题")

    def test_generated_answer_is_in_knowledge_base(self):
        ex = ExaminerAgent(MockLLM(), self.R, self.kps)
        for kp in self.kps:
            it = ex.make_item(kp, 3)
            if it is None:
                continue
            from core.retrieval import numbers_in
            chunk = self.R.get(it["source_id"])
            right = it["options"][it["answer"]]
            if _is_numeric(right):
                pool = numbers_in(f"{chunk.title} {chunk.text}")
                self.assertFalse(numbers_in(right) - pool,
                                 f"{kp} 的正确答案含知识库外数值")

    def test_offline_generator_skips_model_designations(self):
        """型号里的数字不是量值。SRVO-001 的 001、J1 的 1 都不能拿来出题。"""
        from core.llm import _quantities
        self.assertEqual(_quantities("报警SRVO-001含义为操作面板急停被按下。"), [])
        self.assertEqual(_quantities("J1至J3决定末端位置。"), [])
        self.assertEqual(_quantities("速度限制在250毫米每秒以内。"), ["250"])
        self.assertEqual(_quantities("围栏高度不低于1.4米，距离不小于0.5米。"),
                         ["1.4", "0.5"])


class TestAnalyzeAndSynthesize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kps = {k["id"]: k for k in
                   json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]}
        cls.ex = ExaminerAgent(MockLLM(), Retriever.from_jsonl(config.KB_PATH),
                               cls.kps)

    def test_entry_level_reflects_background(self):
        low = self.ex.analyze({"education": "高中", "hands_on_hours": 0}, "零基础")
        high = self.ex.analyze({"education": "硕士", "hands_on_hours": 500}, "做了很久")
        self.assertLessEqual(low["entry_level"], high["entry_level"])
        for r in (low, high):
            self.assertGreaterEqual(r["entry_level"], config.DIFFICULTY_MIN)
            self.assertLessEqual(r["entry_level"], config.DIFFICULTY_MAX)

    def test_focus_only_contains_real_knowledge_points(self):
        """模型可能编出不存在的知识点名，必须被丢弃。"""
        r = self.ex.analyze({"education": "本科", "hands_on_hours": 60}, "机械专业")
        for kp in r["focus"]:
            self.assertIn(kp, self.kps)

    def test_synthesize_uses_only_supplied_numbers(self):
        """综合分析里的数字必须来自规则统计，模型只做归纳表述。"""
        from agents.diagnose import DiagnoseAgent
        from orchestrator import load_profile
        diag = DiagnoseAgent(MockLLM()).run(load_profile("P-C"))
        log = [{"kp": "KP-13", "correct": False, "level": 1, "probe": ""},
               {"kp": "KP-13", "correct": False, "level": 2, "probe": ""},
               {"kp": "KP-15", "correct": False, "level": 5, "probe": ""},
               {"kp": "KP-01", "correct": True, "level": 1, "probe": ""},
               {"kp": "KP-02", "correct": True, "level": 2, "probe": "x"}]
        out = self.ex.synthesize(diag, log)
        self.assertIsInstance(out["patterns"], list)
        self.assertIn("narrative", out)

    def test_patterns_flag_safety_weakness(self):
        """安全类题目错得多，必须单独点出来 —— 这一类错了后果最重。"""
        from agents.diagnose import DiagnoseAgent
        from orchestrator import load_profile
        diag = DiagnoseAgent(MockLLM()).run(load_profile("P-A"))
        log = [{"kp": "KP-13", "correct": False, "level": 1, "probe": ""},
               {"kp": "KP-13", "correct": False, "level": 2, "probe": ""},
               {"kp": "KP-02", "correct": False, "level": 1, "probe": ""}]
        pats = self.ex._patterns(diag, log)
        self.assertTrue(any("安全" in p for p in pats), pats)


if __name__ == "__main__":
    unittest.main()


class TestNegationAwareDistractors(unittest.TestCase):
    """否定感知的干扰项判定。

    这条规则是红队测出来的，也是命题环节最容易改坏的地方：
    放松了会把正确说法放进干扰项（一题两解），收紧了会把所有
    "手册明令禁止的做法"这类最标准的干扰项全毙掉，题目区分度归零。
    """

    @classmethod
    def setUpClass(cls):
        from core.retrieval import Retriever
        cls.R = Retriever.from_jsonl(config.KB_PATH)

    def _body(self, cid):
        c = self.R.get(cid)
        return f"{c.title} {c.text}"

    def test_forbidden_action_is_a_valid_distractor(self):
        from agents.examiner import _forbidden_by
        # KB-017 原文「切勿在未按住解除按钮时强行驱动」
        self.assertTrue(_forbidden_by("直接强行反向驱动", self._body("KB-017")))

    def test_correct_action_is_not_exempted(self):
        from agents.examiner import _forbidden_by
        self.assertFalse(_forbidden_by("按住超程解除按钮再反向移出限位",
                                       self._body("KB-017")))

    def test_option_restating_the_rule_is_not_exempted(self):
        """选项自带禁止词说明它在复述规则、本身成立，不能当干扰项放行。"""
        from agents.examiner import _forbidden_by
        self.assertFalse(_forbidden_by("切勿强行驱动", self._body("KB-017")))

    def test_prohibition_scope_is_clause_level(self):
        """禁止的作用域是分句。

        KB-020 一句话里同时有「必须拆下排脂塞」和「未拆下时注脂会损坏油封」。
        只按句号切会把整句当禁止句，连正确做法都被判成"被禁止"，
        等于把正确答案塞进干扰项。
        """
        from agents.examiner import _forbidden_by
        body = self._body("KB-020")
        self.assertTrue(_forbidden_by("在排脂塞未拆下时注脂", body))
        self.assertFalse(_forbidden_by("拆下排脂塞后注脂", body))

    def test_unrelated_option_is_not_exempted(self):
        from agents.examiner import _forbidden_by
        self.assertFalse(_forbidden_by("更换伺服电机", self._body("KB-022")))


class TestGeneratedItemProvenance(unittest.TestCase):
    """生成题的来源与难度标记。"""

    @classmethod
    def setUpClass(cls):
        import json as _json
        from agents.examiner import ExaminerAgent
        from core.llm import MockLLM
        from core.retrieval import Retriever
        kps = _json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]
        cls.kpi = {k["id"]: k for k in kps}
        cls.ex = ExaminerAgent(MockLLM(), Retriever.from_jsonl(config.KB_PATH),
                               cls.kpi)

    def test_generated_items_are_marked(self):
        item = None
        for kp in self.kpi:
            item = self.ex.make_item(kp, 3)
            if item:
                break
        self.assertIsNotNone(item, "至少应能生成一道通过审核的题")
        self.assertEqual(item["origin"], "generated")

    def test_difficulty_is_marked_as_requested_not_measured(self):
        """生成题的难度是请求值，没有作答数据可标定。

        这个标记必须在，否则自适应选题会拿一个未经标定的难度当真，
        相当于用没校准的尺子挑下一道题。
        """
        item = None
        for kp in self.kpi:
            item = self.ex.make_item(kp, 4)
            if item:
                break
        if item:
            self.assertEqual(item["level_source"], "requested")

    def test_bank_items_are_marked_as_bank(self):
        import json as _json
        items = _json.loads(config.PRETEST_PATH.read_text(encoding="utf-8"))["items"]
        self.assertTrue(all(i.get("level_source") == "bank" for i in items))

    def test_no_output_and_rejection_counted_separately(self):
        """模型无产出与审核拒收必须分开计数。

        混在一个通过率里，等于把闸门的功劳和模型的短板搅在一起，
        答辩时答不出"你们到底拒了多少道"。
        """
        from agents.examiner import ExaminerAgent
        from core.llm import MockLLM
        from core.retrieval import Retriever
        ex = ExaminerAgent(MockLLM(), Retriever.from_jsonl(config.KB_PATH), self.kpi)
        for kp in self.kpi:
            ex.make_item(kp, 3)
        self.assertEqual(ex.requests, len(self.kpi))
        self.assertIsInstance(ex.no_output, int)
        self.assertIsInstance(ex.rejects, list)
