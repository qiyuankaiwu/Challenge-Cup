"""Python 后端与浏览器端引擎的一致性测试。

web/engine.js 把检索、BKT、自适应选题、辩论、审核重新实现了一遍，
好让离线的 showcase.html 能真跑而不是回放录像。两套实现一旦漂移，
演示给出的结论就和后端对不上 —— 这在答辩现场是致命的。

所以用同一批输入分别跑两边，逐项比对。任何一边改了规则没同步，这里会红。

需要 node。没装 node 时整体跳过，不阻塞其他测试。
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "web" / "engine.js"
SNAPSHOT = ROOT / "web" / "snapshot.json"
NODE = shutil.which("node")


def run_js(script: str) -> dict:
    """在 node 里加载 engine.js 并执行一段脚本，返回它打印的 JSON。"""
    harness = f"""
    global.window = global;
    const fs = require('fs');
    eval(fs.readFileSync({str(ENGINE)!r}, 'utf8'));
    const SNAP = JSON.parse(fs.readFileSync({str(SNAPSHOT)!r}, 'utf8'));
    {script}
    """
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(harness)
        path = fh.name
    out = subprocess.run(
        [NODE, path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if out.returncode != 0:
        raise AssertionError(f"node 执行失败：{out.stderr[:600]}")
    return json.loads(out.stdout)


@unittest.skipUnless(NODE and SNAPSHOT.exists(), "需要 node 与 web/snapshot.json")
class TestParity(unittest.TestCase):

    def test_tokenizer_and_overlap(self):
        from core.retrieval import overlap_ratio, tokenize
        pairs = [
            ("报警SRVO-005含义为机器人超程。", "报警SRVO-005含义为机器人超程。处理方法是按住超程解除按钮。"),
            ("T1模式下末端速度限制在250毫米每秒以内。", "机器人运行模式分为手动低速T1、手动高速T2和自动AUTO三种。"),
            ("加注润滑脂时必须打开排脂口。", "加注润滑脂时必须打开排脂口，禁止在排脂口封闭状态下加注。"),
        ]
        js = run_js(f"""
        const P = {json.dumps(pairs, ensure_ascii=False)};
        console.log(JSON.stringify(P.map(([a,b]) => ({{
          n: Engine.tokenize(a).length,
          r: Math.round(Engine.overlapRatio(a,b)*1000)/1000
        }}))));
        """)
        for (a, b), got in zip(pairs, js):
            self.assertEqual(got["n"], len(tokenize(a)), f"分词数不一致：{a}")
            self.assertAlmostEqual(got["r"], round(overlap_ratio(a, b), 3), places=3,
                                   msg=f"覆盖率不一致：{a}")

    def test_chinese_numerals(self):
        from core.retrieval import cn_to_int, numbers_in
        words = ["一万", "两万", "三千", "二十五", "十", "一亿", "八"]
        texts = ["润滑脂更换周期为运行一万小时或三年",
                 "T1模式限速250毫米每秒", "子程序不超过8层"]
        js = run_js(f"""
        const W = {json.dumps(words, ensure_ascii=False)};
        const T = {json.dumps(texts, ensure_ascii=False)};
        console.log(JSON.stringify({{
          w: W.map(x => Engine.cnToInt(x)),
          t: T.map(x => [...Engine.numbersIn(x)].sort())
        }}));
        """)
        self.assertEqual(js["w"], [cn_to_int(w) for w in words])
        for text, got in zip(texts, js["t"]):
            self.assertEqual(got, sorted(numbers_in(text)), f"数值集合不一致：{text}")

    def test_bkt_update_matches(self):
        from core import bkt
        p = bkt.BKTParams(p_T=config.BKT_P_T, p_S=config.BKT_P_S, p_G=config.BKT_P_G)
        seqs = [[True], [False], [True, True, False],
                [False, False, True, True], [True] * 5]
        js = run_js(f"""
        const S = {json.dumps(seqs)};
        const P = {{p_T:{config.BKT_P_T}, p_S:{config.BKT_P_S}, p_G:{config.BKT_P_G}}};
        console.log(JSON.stringify(S.map(seq => {{
          let p = 0.2;
          for (const c of seq) p = Engine.BKT.update(p, c, P);
          return Math.round(p*10000)/10000;
        }})));
        """)
        for seq, got in zip(seqs, js):
            want = round(bkt.trace(seq, p, p_L0=0.2), 4)
            self.assertAlmostEqual(got, want, places=4, msg=f"BKT 不一致：{seq}")

    def test_prior_matches(self):
        from core import bkt
        bgs = [
            {"education": "本科", "hands_on_hours": 0},
            {"education": "高职", "hands_on_hours": 120},
            {"education": "硕士", "hands_on_hours": 480},
            {"education": "", "hands_on_hours": 0},
        ]
        js = run_js(f"""
        const B = {json.dumps(bgs, ensure_ascii=False)};
        console.log(JSON.stringify(B.map(b => Math.round(Engine.BKT.prior(b)*10000)/10000)));
        """)
        for bg, got in zip(bgs, js):
            self.assertAlmostEqual(got, round(bkt.prior_from_background(bg), 4),
                                   places=4, msg=f"先验不一致：{bg}")

    def test_intake_parsing_matches(self):
        from agents.intake import rule_extract
        texts = [
            "我是机械专业大三的，实操大概40小时",
            "高职电气自动化二年级，实训带过搬运工作站，前后三个月",
            "在装配线做了六年，转岗做机器人看护，零基础",
            "硕士在读，做视觉算法，机器人完全没碰过，想补现场调试",
        ]
        js = run_js(f"""
        const T = {json.dumps(texts, ensure_ascii=False)};
        console.log(JSON.stringify(T.map(t => {{
          const r = Engine.parseIntake(t);
          return {{education:r.education, grade:r.grade, major:r.major,
                   hands_on_hours:r.hands_on_hours}};
        }})));
        """)
        for text, got in zip(texts, js):
            want = rule_extract(text)
            for k in ("education", "grade", "major", "hands_on_hours"):
                self.assertEqual(got[k], want[k], f"字段 {k} 不一致：{text}")

    def test_audit_verdicts_match(self):
        from agents.audit import AuditAgent
        from core.llm import MockLLM
        from core.retrieval import Retriever
        from core.schema import Claim
        cases = [
            ("报警SRVO-005含义为机器人超程。", "KB-017"),
            ("T1模式下末端法兰中心的移动速度被限制在200毫米每秒以内。", "KB-004"),
            ("机器人安全围栏高度不低于1.4米。", "KB-022"),
            ("报警SRVO-002含义为机器人超程。", "KB-016"),
            ("减速机润滑脂应当每运行三千小时更换一次。", "KB-022"),
            ("控制柜每运行500小时需要更换一次主控板电池。", None),
            ("工件坐标系可通过六点法标定。", "KB-999"),
        ]
        auditor = AuditAgent(MockLLM(), Retriever.from_jsonl(config.KB_PATH))
        js = run_js(f"""
        const R = new Engine.Retriever(SNAP.kb);
        const C = {json.dumps([[t, s] for t, s in cases], ensure_ascii=False)};
        console.log(JSON.stringify(C.map(([text, src]) =>
          Engine.auditOne({{text, source_id:src}}, R).verdict)));
        """)
        for (text, src), got in zip(cases, js):
            kept, dropped = auditor.review([Claim(text=text, source_id=src)])
            want = (kept + dropped)[0].verdict
            self.assertEqual(got, want, f"审核判定不一致：{text}")

    def test_adaptive_selection_order_matches(self):
        """同一份背景、同一串对错，两边选出的题号序列必须完全一样。"""
        from core.cat import AdaptiveSession
        items = json.loads(config.PRETEST_PATH.read_text(encoding="utf-8"))["items"]
        kps = json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]
        bg = {"education": "高职", "hands_on_hours": 480}
        pattern = [True, False, True, True, False, False, True, True,
                   False, True, False, True, True, False, True, False]

        s = AdaptiveSession(items, kps, bg, max_items=16)
        order_py, log_py = [], []
        while True:
            it = s.next_item()
            if it is None:
                break
            ok = pattern[len(order_py) % len(pattern)]
            order_py.append(it["id"])
            step = s.answer(it["id"], it["answer"] if ok
                            else (it["answer"] + 1) % len(it["options"]))
            log_py.append((round(step["after"], 4), step["probe"] != ""))

        js = run_js(f"""
        const bg = {json.dumps(bg, ensure_ascii=False)};
        const pat = {json.dumps(pattern)};
        const a = new Engine.Adaptive(SNAP.items, SNAP.kps, bg, {{maxItems:16}});
        const order = [], log = [];
        for (;;) {{
          const it = a.next();
          if (!it) break;
          const ok = pat[order.length % pat.length];
          order.push(it.id);
          const st = a.answer(it.id, ok ? it.answer : (it.answer+1) % it.options.length);
          log.push([Math.round(st.after*10000)/10000, st.probe !== ""]);
        }}
        console.log(JSON.stringify({{order, log}}));
        """)
        self.assertEqual(js["order"], order_py, "选题顺序不一致")
        self.assertEqual([tuple(x) for x in js["log"]], log_py, "掌握概率或追问判定不一致")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(NODE and SNAPSHOT.exists(), "需要 node 与 web/snapshot.json")
class TestExaminerParity(unittest.TestCase):
    """命题环节的两端一致性。

    命题审核是后果最重的一关：题目答案错了，系统会拿一把错的尺子量人。
    两端判据一旦漂移，浏览器里放行的题在后端会被毙掉，反之亦然，
    同一个学习者在两处会得到不同的掌握度。
    """

    def test_quantity_extraction_matches(self):
        from core.llm import _quantities
        texts = ["报警SRVO-001含义为操作面板急停被按下。",
                 "J1至J3决定末端位置，共六个关节轴。",
                 "移动速度被限制在250毫米每秒以内。",
                 "围栏高度不低于1.4米，距离不小于0.5米。",
                 "子程序调用层数一般不超过8层。"]
        js = run_js(f"""
        const T = {json.dumps(texts, ensure_ascii=False)};
        console.log(JSON.stringify(T.map(t => Engine.quantities(t))));
        """)
        for text, got in zip(texts, js):
            self.assertEqual(got, _quantities(text), f"量值抽取不一致：{text}")

    def test_numeric_option_detection_matches(self):
        from agents.examiner import _is_numeric
        opts = ["0.5", "250", "1.4米", "8层", "三年", "一万小时",
                "不小于30度", "按住超程解除按钮", ""]
        js = run_js(f"""
        const O = {json.dumps(opts, ensure_ascii=False)};
        console.log(JSON.stringify(O.map(o => Engine.isNumericOption(o))));
        """)
        for o, got in zip(opts, js):
            self.assertEqual(got, _is_numeric(o), f"数值型判定不一致：{o!r}")

    def test_item_vetting_matches(self):
        """同一批题目，两端的放行/驳回结论必须一致。"""
        from agents.examiner import ExaminerAgent, ItemRejected
        from core.llm import MockLLM
        from core.retrieval import Retriever
        kps = {k["id"]: k for k in
               json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]}
        ex = ExaminerAgent(MockLLM(), Retriever.from_jsonl(config.KB_PATH), kps)

        items = [
            # 合格
            {"stem": "安全围栏高度要求是", "options": ["1.4", "2.8", "0.7", "5.6"],
             "answer": 0, "source_id": "KB-022"},
            # 答案无依据
            {"stem": "安全围栏高度要求是", "options": ["9.9", "2.8", "0.7", "5.6"],
             "answer": 0, "source_id": "KB-022"},
            # 干扰项也成立
            {"stem": "安全围栏高度要求是", "options": ["1.4", "0.5", "2.8", "5.6"],
             "answer": 0, "source_id": "KB-022"},
            # 题干凭空出现数值
            {"stem": "依据第 37 条，安全围栏高度要求是",
             "options": ["1.4", "2.8", "0.7", "5.6"], "answer": 0,
             "source_id": "KB-022"},
            # 长度线索
            {"stem": "超程报警的处理方法是",
             "options": ["按住超程解除按钮，同时在关节坐标系下手动将超程轴反向移出限位范围",
                         "喷漆", "扫码", "联网"],
             "answer": 0, "source_id": "KB-017"},
            # 引用悬空
            {"stem": "安全围栏高度要求是", "options": ["1.4", "2.8", "0.7", "5.6"],
             "answer": 0, "source_id": "KB-999"},
            # 选项重复
            {"stem": "安全围栏高度要求是", "options": ["1.4", "1.4", "0.7", "5.6"],
             "answer": 0, "source_id": "KB-022"},
        ]
        js = run_js(f"""
        const R = new Engine.Retriever(SNAP.kb);
        const I = {json.dumps(items, ensure_ascii=False)};
        console.log(JSON.stringify(I.map(it => Engine.vetItem(it, R) === null)));
        """)
        for item, got in zip(items, js):
            try:
                ex.vet({**item, "kp": "KP-13", "level": 2})
                want = True
            except ItemRejected:
                want = False
            self.assertEqual(got, want,
                             f"审核结论不一致：{item['stem'][:20]} 期望通过={want}")


@unittest.skipUnless(NODE and SNAPSHOT.exists(), "需要 node 与 web/snapshot.json")
class TestEvidenceParity(unittest.TestCase):
    """证据强度的两端一致性。

    区间与蒙对概率直接决定界面上写的是"已确认掌握"还是"疑似掌握"。
    两端漂移的话，离线演示给出的结论会和后端评测对不上，
    这在答辩现场是致命的。
    """

    def test_luck_probability_matches(self):
        from core import bkt
        cases = [(1, 1), (2, 2), (3, 3), (3, 4), (4, 4), (5, 6), (0, 2), (2, 5)]
        js = run_js(f"""
        const C = {json.dumps(cases)};
        console.log(JSON.stringify(C.map(([k,n]) =>
          Math.round(Engine.luckProbability(k,n,0.25)*100000)/100000)));
        """)
        for (k, n), got in zip(cases, js):
            self.assertAlmostEqual(got, round(bkt.luck_probability(k, n), 5),
                                   places=4, msg=f"{k}/{n} 蒙对概率不一致")

    def test_mastery_interval_matches(self):
        from core import bkt
        p = bkt.BKTParams(p_T=config.BKT_P_T, p_S=config.BKT_P_S, p_G=config.BKT_P_G)
        cases = [(1, 1), (2, 2), (3, 3), (3, 4), (4, 4), (5, 6), (0, 2), (8, 8)]
        js = run_js(f"""
        const C = {json.dumps(cases)};
        const P = {{p_T:{config.BKT_P_T}, p_S:{config.BKT_P_S}, p_G:{config.BKT_P_G}}};
        console.log(JSON.stringify(C.map(([k,n]) => Engine.masteryInterval(k,n,P))));
        """)
        for (k, n), got in zip(cases, js):
            want = bkt.mastery_interval(k, n, p)
            self.assertAlmostEqual(got[0], want[0], places=2, msg=f"{k}/{n} 下界不一致")
            self.assertAlmostEqual(got[1], want[1], places=2, msg=f"{k}/{n} 上界不一致")

    def test_evidence_state_matches(self):
        from core import bkt
        p = bkt.BKTParams(p_T=config.BKT_P_T, p_S=config.BKT_P_S, p_G=config.BKT_P_G)
        cases = [(1, 1), (2, 2), (3, 4), (4, 4), (10, 10), (0, 1), (0, 2), (2, 4)]
        js = run_js(f"""
        const C = {json.dumps(cases)};
        const P = {{p_T:{config.BKT_P_T}, p_S:{config.BKT_P_S}, p_G:{config.BKT_P_G}}};
        console.log(JSON.stringify(C.map(([k,n]) => {{
          let pL = 0.3;
          for (let i=0;i<n;i++) pL = Engine.BKT.update(pL, i<k, P);
          const iv = Engine.masteryInterval(k,n,P);
          const lk = n ? Engine.luckProbability(k,n,P.p_G) : 1;
          return Engine.evidenceState(pL, iv[0], lk, n,
                    {config.MASTERY_OK}, {config.MASTERY_BLIND})[0];
        }})));
        """)
        for (k, n), got in zip(cases, js):
            obs = [True] * k + [False] * (n - k)
            score = bkt.trace(obs, p, p_L0=0.3) if n else 0.0
            lo, _ = bkt.mastery_interval(k, n, p)
            lk = bkt.luck_probability(k, n) if n else 1.0
            want, _ = bkt.evidence_state(score, lo, lk, n,
                                         config.MASTERY_OK, config.MASTERY_BLIND)
            self.assertEqual(got, want, f"{k}/{n} 判定不一致")

    def test_ability_profile_matches(self):
        from agents.diagnose import DiagnoseAgent
        from core.ability import build
        from core.llm import MockLLM
        from orchestrator import load_profile
        for pid in ("P-A", "P-B", "P-C"):
            diag = DiagnoseAgent(MockLLM()).run(load_profile(pid))
            want = build(diag)
            js = run_js(f"""
            const d = SNAP.sessions[{pid!r}].diagnosis;
            const p = Engine.buildAbility(d, SNAP.dims);
            console.log(JSON.stringify(p.dims.map(x => [x.name, x.score, x.lower])));
            """)
            for got, w in zip(js, want.dims):
                self.assertEqual(got[0], w.name)
                self.assertAlmostEqual(got[1], w.score, places=2,
                                       msg=f"{pid} {w.name} 点估计不一致")
                self.assertAlmostEqual(got[2], w.lower, places=2,
                                       msg=f"{pid} {w.name} 下界不一致")
