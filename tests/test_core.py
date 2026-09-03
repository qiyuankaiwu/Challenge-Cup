"""检索、学情诊断、决策规则的单元测试。"""

import unittest

from agents.decide import ACTION_DOWN, ACTION_HOLD, ACTION_UP, DecideAgent
from agents.diagnose import DiagnoseAgent
from agents.generate import learner_level
import config
from core.llm import MockLLM
from core.retrieval import Retriever, overlap_ratio, tokenize
from orchestrator import load_profile


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Retriever.from_jsonl(config.KB_PATH)

    def test_model_codes_survive_tokenization(self):
        toks = tokenize("报警SRVO-062需要更换电池")
        self.assertIn("srvo", toks)
        self.assertIn("062", toks)

    def test_topk_hits_expected_chunk(self):
        hits = self.r.search("急停按钮被按下如何复位", top_k=3)
        ids = [c.id for c, _ in hits]
        self.assertTrue({"KB-015", "KB-016"} & set(ids), ids)

    def test_kp_filter_is_hard(self):
        hits = self.r.search("坐标系", top_k=5, kp="KP-10")
        self.assertTrue(all(c.kp == "KP-10" for c, _ in hits))

    def test_every_kp_has_at_least_one_chunk(self):
        """知识库不能有空知识点，否则覆盖率指标会在那里掉下去。"""
        import json
        kps = json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]
        for kp in kps:
            self.assertTrue(self.r.by_kp(kp["id"]),
                            f"{kp['id']} {kp['name']} 没有对应切片")

    def test_overlap_ratio_bounds(self):
        self.assertEqual(overlap_ratio("", "任意"), 0.0)
        self.assertAlmostEqual(overlap_ratio("安全围栏", "安全围栏"), 1.0)


class TestDiagnose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = DiagnoseAgent(MockLLM())

    def test_scoring_is_deterministic(self):
        p = load_profile("P-A")
        a, b = self.agent.run(p), self.agent.run(p)
        self.assertEqual([m.score for m in a.mastery], [m.score for m in b.mastery])
        self.assertEqual(a.gaps, b.gaps)

    def test_profiles_produce_different_gaps(self):
        gaps = {pid: tuple(self.agent.run(load_profile(pid)).gaps)
                for pid in ("P-A", "P-B", "P-C")}
        self.assertEqual(len(set(gaps.values())), 3, "三个画像的盲区应当各不相同")

    def test_gap_order_respects_prerequisites(self):
        diag = self.agent.run(load_profile("P-C"))
        pos = {kp: i for i, kp in enumerate(diag.gaps)}
        for kp in diag.gaps:
            for pre in self.agent._kp_index[kp]["prereq"]:
                if pre in pos:
                    self.assertLess(pos[pre], pos[kp],
                                    f"{pre} 是 {kp} 的前置，却排在后面")

    def test_weaker_learner_gets_lower_entry_level(self):
        a = self.agent.run(load_profile("P-A"))
        c = self.agent.run(load_profile("P-C"))
        self.assertLessEqual(c.entry_level, a.entry_level)

    def test_learner_level_within_bounds(self):
        diag = self.agent.run(load_profile("P-B"))
        for m in diag.mastery:
            self.assertGreaterEqual(learner_level(m), config.DIFFICULTY_MIN)
            self.assertLessEqual(learner_level(m), config.DIFFICULTY_MAX)


class TestDecide(unittest.TestCase):
    def setUp(self):
        self.agent = DecideAgent()

    def test_low_accuracy_triggers_downshift(self):
        d = self.agent.run("KP-04", [False, False, True, False], 3)
        self.assertEqual(d["action"], ACTION_DOWN)
        self.assertEqual(d["next_difficulty"], 2)

    def test_high_accuracy_triggers_advance(self):
        d = self.agent.run("KP-04", [True] * 4, 3)
        self.assertEqual(d["action"], ACTION_UP)
        self.assertEqual(d["next_difficulty"], 4)

    def test_middle_band_holds(self):
        d = self.agent.run("KP-04", [True, True, True, False], 3)
        self.assertEqual(d["action"], ACTION_HOLD)
        self.assertEqual(d["next_difficulty"], 3)

    def test_small_sample_does_not_move(self):
        d = self.agent.run("KP-04", [False], 3)
        self.assertEqual(d["action"], ACTION_HOLD)
        self.assertIsNone(d["accuracy"])

    def test_difficulty_never_leaves_range(self):
        self.assertEqual(self.agent.run("KP-01", [False] * 3, 1)["next_difficulty"], 1)
        self.assertEqual(self.agent.run("KP-15", [True] * 3, 5)["next_difficulty"], 5)

    def test_decision_is_reproducible(self):
        answers = [True, False, True, True, False]
        first = self.agent.run("KP-06", answers, 3)
        for _ in range(20):
            self.assertEqual(self.agent.run("KP-06", answers, 3), first)


if __name__ == "__main__":
    unittest.main()


class TestBKT(unittest.TestCase):
    """BKT 估计器的性质测试。

    这些测试不是查具体数值，是查性质：单调性、退化检测、蒙对失误的处理方向。
    参数以后重新拟合了，这些性质仍然必须成立。
    """

    def setUp(self):
        from core import bkt
        self.bkt = bkt
        self.p = bkt.BKTParams(p_T=config.BKT_P_T, p_S=config.BKT_P_S,
                               p_G=config.BKT_P_G)

    def test_degenerate_params_rejected(self):
        bad = self.bkt.BKTParams(p_S=0.6, p_G=0.5)
        with self.assertRaises(ValueError):
            bad.validate()

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            self.bkt.BKTParams(p_S=1.4).validate()

    def test_correct_raises_mastery(self):
        self.assertGreater(self.bkt.update(0.4, True, self.p), 0.4)

    def test_wrong_lowers_mastery(self):
        self.assertLess(self.bkt.update(0.6, False, self.p), 0.6)

    def test_single_lucky_guess_does_not_certify(self):
        """蒙对一道题不能算掌握。答对率会给 100%，BKT 不会。"""
        p = self.bkt.trace([True], self.p, p_L0=0.2)
        self.assertLess(p, config.MASTERY_OK)

    def test_one_slip_does_not_destroy_a_strong_learner(self):
        """连对四题后错一题，不应被打成盲区。"""
        p = self.bkt.trace([True, True, True, True, False], self.p, p_L0=0.3)
        self.assertGreater(p, config.MASTERY_BLIND)

    def test_monotone_in_number_correct(self):
        seqs = [[False] * 4, [True] + [False] * 3, [True] * 2 + [False] * 2,
                [True] * 3 + [False], [True] * 4]
        vals = [self.bkt.trace(s, self.p, p_L0=0.2) for s in seqs]
        self.assertEqual(vals, sorted(vals))

    def test_prior_reflects_background(self):
        novice = self.bkt.prior_from_background(
            {"education": "高中", "hands_on_hours": 0})
        veteran = self.bkt.prior_from_background(
            {"education": "本科在读", "hands_on_hours": 120})
        self.assertLess(novice, veteran)
        self.assertTrue(0 < novice < 1 and 0 < veteran < 1)

    def test_confidence_grows_with_evidence(self):
        vals = [self.bkt.confidence(n) for n in range(0, 6)]
        self.assertEqual(vals, sorted(vals))
        self.assertEqual(vals[0], 0.0)

    def test_curve_length_matches_observations(self):
        curve = self.bkt.trace_curve([True, False, True], self.p, p_L0=0.3)
        self.assertEqual(len(curve), 4)


class TestChineseNumerals(unittest.TestCase):
    """中文数字归一化。红队测试逼出来的，专治 H3 类数值篡改。"""

    def test_basic_conversion(self):
        from core.retrieval import cn_to_int
        cases = {"一万": 10000, "两万": 20000, "三千": 3000, "二十五": 25,
                 "十": 10, "一亿": 100000000, "八": 8}
        for text, want in cases.items():
            self.assertEqual(cn_to_int(text), want, text)

    def test_mismatch_is_detectable(self):
        from core.retrieval import numbers_in
        src = numbers_in("润滑脂更换周期为运行一万小时或三年")
        bad = numbers_in("润滑脂更换周期为运行两万小时或五年")
        self.assertTrue(bad - src, "篡改后的数值应当能被检出为新增数字")

    def test_arabic_and_chinese_normalize_together(self):
        from core.retrieval import numbers_in
        self.assertIn("10000", numbers_in("一万小时"))
        self.assertIn("10000", numbers_in("10000小时"))


class TestAdaptation(unittest.TestCase):
    """适配判定的边界。重点是知识点难度天花板那一条。"""

    def setUp(self):
        from agents.generate import is_adapted
        self.f = is_adapted

    def test_inside_window_is_adapted(self):
        self.assertTrue(self.f(3, 3, 4))
        self.assertTrue(self.f(5, 3, 5))

    def test_above_window_is_not(self):
        self.assertFalse(self.f(5, 2, 5))

    def test_below_window_is_not_when_ceiling_allows(self):
        self.assertFalse(self.f(2, 3, 5))

    def test_ceiling_case_counts_as_adapted(self):
        """一级知识点对上被估为二级的薄弱学习者：只有一级内容可讲，算适配。

        这是演示界面顶出来的规格漏洞。原判定不看知识点上限，
        把这种无解约束记成"不适配"，等于惩罚系统做对的事。
        """
        self.assertTrue(self.f(1, 2, 1, strong=False))

    def test_strong_learner_gets_one_extra_level(self):
        self.assertTrue(self.f(2, 3, 1, strong=True))
        self.assertFalse(self.f(1, 3, 1, strong=True))

    def test_ceiling_never_exceeds_max(self):
        self.assertTrue(self.f(5, 5, 5, strong=True))


class TestAdaptiveSelection(unittest.TestCase):
    """自适应测评的行为。"""

    def setUp(self):
        import json
        from core.cat import AdaptiveSession
        self.items = json.loads(config.PRETEST_PATH.read_text(encoding="utf-8"))["items"]
        self.kps = json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]
        self.make = lambda bg, n=16: AdaptiveSession(self.items, self.kps, bg, max_items=n)

    def _run(self, s, rule):
        """rule(item) -> bool，决定这道题答对还是答错。"""
        while True:
            it = s.next_item()
            if it is None:
                return s
            ok = rule(it)
            s.answer(it["id"], it["answer"] if ok
                     else (it["answer"] + 1) % len(it["options"]))

    def test_never_exceeds_budget(self):
        s = self._run(self.make({"education": "本科"}, 8), lambda it: True)
        self.assertLessEqual(len(s.asked), 8)

    def test_never_repeats_an_item(self):
        s = self._run(self.make({"education": "高职", "hands_on_hours": 100}),
                      lambda it: it["level"] <= 3)
        self.assertEqual(len(s.asked), len(set(s.asked)))

    def test_background_changes_prior(self):
        weak = self.make({"education": "高中", "hands_on_hours": 0})
        strong = self.make({"education": "硕士", "hands_on_hours": 480})
        self.assertLess(weak.prior, strong.prior)

    def test_all_correct_yields_no_blind(self):
        s = self._run(self.make({"education": "本科"}, 20), lambda it: True)
        from agents.diagnose import DiagnoseAgent
        for kp, st in s.state.items():
            if st["n"] > 0:
                self.assertGreater(st["p"], config.MASTERY_BLIND, kp)

    def test_all_wrong_drives_mastery_down(self):
        s = self._run(self.make({"education": "硕士", "hands_on_hours": 480}, 20),
                      lambda it: False)
        for kp, st in s.state.items():
            if st["n"] > 0:
                self.assertLess(st["p"], config.MASTERY_WEAK, kp)

    def test_prereq_inference_skips_downstream(self):
        """前置确认盲区后，后继知识点不该再占题量。"""
        s = self._run(self.make({"education": "高中", "hands_on_hours": 0}, 20),
                      lambda it: False)
        blocked = s.blocked_by_prereq()
        self.assertTrue(blocked, "全错的情况下应当推断出若干后继盲区")
        for kp, pre in blocked.items():
            self.assertIn(pre, s.kps[kp]["prereq"])

    def test_probe_requires_prior_evidence(self):
        """第一道题永远不触发追问 —— 那时没有证据可冲突。"""
        s = self.make({"education": "高中", "hands_on_hours": 0})
        it = s.next_item()
        step = s.answer(it["id"], it["answer"])
        self.assertEqual(step["probe"], "")

    def test_probe_fires_on_conflicting_evidence(self):
        """同一知识点先连对再答错，应触发追问。"""
        from core.cat import AdaptiveSession
        one_kp = [i for i in self.items if i["kp"] == "KP-10"]
        s = AdaptiveSession(one_kp, [k for k in self.kps if k["id"] == "KP-10"],
                            {"education": "本科", "hands_on_hours": 100}, max_items=6)
        it = s.next_item(); s.answer(it["id"], it["answer"])
        it = s.next_item(); s.answer(it["id"], it["answer"])
        it = s.next_item()
        step = s.answer(it["id"], (it["answer"] + 1) % len(it["options"]))
        self.assertTrue(step["probe"], "连对后答错应当触发追问")

    def test_responses_replay_matches_diagnosis(self):
        """测评导出的 responses 交给 DiagnoseAgent 复算，掌握度必须一致。

        两条路径算出来不一样的话，界面显示的和后端评测的就是两回事。
        """
        from agents.diagnose import DiagnoseAgent
        from core.llm import MockLLM
        bg = {"education": "高职", "hands_on_hours": 480}
        s = self._run(self.make(bg, 20), lambda it: it["level"] <= 3)
        profile = {"id": "T", "background": bg, "responses": s.responses()}
        diag = DiagnoseAgent(MockLLM()).run(profile)
        for m in diag.mastery:
            if m.asked > 0:
                self.assertAlmostEqual(m.score, round(s.state[m.kp]["p"], 3),
                                       places=3, msg=f"{m.kp} 两条路径掌握度不一致")


class TestIntake(unittest.TestCase):
    """自述抽取。"""

    def setUp(self):
        from agents.intake import rule_extract
        self.f = rule_extract

    def test_grade_is_not_work_experience(self):
        """「二年级」是学制不是工龄。不排掉会被算成 3200 小时实操。"""
        r = self.f("高职电气自动化二年级，实训带过搬运工作站")
        self.assertEqual(r["grade"], "二年级")
        self.assertEqual(r["hands_on_hours"], 0)

    def test_duration_units_normalize_to_hours(self):
        self.assertEqual(self.f("实操大概40小时")["hands_on_hours"], 40)
        self.assertEqual(self.f("实训课接触过两小时工业机器人")["hands_on_hours"], 2)
        self.assertEqual(self.f("累计训练二十五小时")["hands_on_hours"], 25)
        self.assertEqual(self.f("体验了半小时")["hands_on_hours"], 0.5)
        self.assertEqual(self.f("练习了一个半小时")["hands_on_hours"], 1.5)
        self.assertEqual(self.f("上手了三个月")["hands_on_hours"], 480)
        self.assertEqual(self.f("练了两周")["hands_on_hours"], 80)

    def test_zero_hint_overrides_duration(self):
        r = self.f("在装配线做了六年，机器人零基础")
        self.assertEqual(r["hands_on_hours"], 0)

    def test_education_levels(self):
        for text, want in [("我是博士在读", "博士"), ("硕士研究生", "硕士"),
                           ("机械专业大三", "本科"), ("高职二年级", "高职"),
                           ("高中毕业就上班了", "高中")]:
            self.assertEqual(self.f(text)["education"], want, text)

    def test_empty_input_is_safe(self):
        r = self.f("")
        self.assertEqual(r["hands_on_hours"], 0)
        self.assertEqual(r["education"], "")

    def test_hours_are_capped(self):
        self.assertLessEqual(self.f("干了二十年产线")["hands_on_hours"], 2000)

    def test_clarify_asks_only_for_missing(self):
        from agents.intake import clarify
        full = {"education": "本科", "hands_on_hours": 40, "major": "机械类",
                "grade": "三年级"}
        self.assertEqual(clarify(full), [])
        self.assertTrue(clarify({"education": "", "hands_on_hours": 0}))
