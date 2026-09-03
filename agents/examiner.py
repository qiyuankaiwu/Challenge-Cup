"""命题 Agent。

固定题库有天花板：47 道题、每个知识点 3 道，难度档位也就那么几级。学习者
水平落在两道题之间时，只能拿一道偏难或偏易的凑合，测出来的掌握度自然糙。

所以让大模型现场命题：读完自述先做一段分析，定下起点难度和重点方向，
再按需要的知识点和难度出题，出的是同一事实的变式（换问法、换考查角度、
换干扰项），不是凭空造知识。

**但命题必须过审，而且比生成讲义更严。**

理由很硬：讲义写错一句，学习者可能看出来；**题目答案错了，系统会拿一把错的
尺子去量人**。答对被判成答错，掌握度往反方向走，后面所有资源推荐跟着错，
而且整条链路上没有任何环节会发现。这是本项目里后果最重的一类错误。

所以审核分四关，全过才录用：
  1. 正确答案必须被所引切片支撑（沿用 AuditAgent 那套判据）
  2. 每个干扰项都必须**不被**支撑 —— 有两个正确答案的题比答案错了还糟
  3. 题干不得引入切片以外的数值
  4. 选项不得重复、不得有明显长度线索（最长的那个是答案是常见出题瑕疵）

过不了就丢，重试有限次，再不行退回固定题库。**宁可用一道糙题，
不能用一道错题。**

另一条：生成题的参数可信度低于人工题。BKT 的蒙对率 p_G 按选项数算，
但生成题的实际区分度未知，所以给它更保守的 p_S，等价于告诉模型
"这道题的信息量打个折"。参数写在 config，别在代码里散落。
"""

from __future__ import annotations

import json
import re

import config
from agents.audit import AuditAgent
from core.llm import parse_json
from core.itemquality import structural
from core.retrieval import Retriever, numbers_in, overlap_ratio
from core.schema import Claim, VERDICT_CONTRADICTED, VERDICT_UNSUPPORTED

_ANALYZE_SYS = (
    "你是职业技能培训的学情分析师。读学习者的自述，判断他的起点。"
    "只输出 JSON："
    '{"summary":"两句话内的分析","entry_level":1到5的整数,'
    '"focus":["知识点名称"],"caution":"需要注意的地方"}。'
    "只依据自述内容判断，不要编造未提到的经历。"
)

_ITEM_SYS = (
    "你是工业机器人领域的命题老师。严格依据给定资料出一道四选一单选题。"
    "要求：正确答案必须能在资料原文中找到依据；三个干扰项必须是资料中"
    "明确错误或未提及的说法；题干和选项不得出现资料以外的数值；"
    "四个选项长度尽量接近。"
    "只输出 JSON："
    '{"stem":"题干","options":["A","B","C","D"],"answer":0到3的整数,'
    '"explain":"依据说明"}'
)

_SYNTH_SYS = (
    "你是学情报告撰写人。给你的是已经算好的测评数据，"
    "请用两到四句话客观描述这位学习者的水平特征和薄弱环节。"
    "不要重复数字，不要给出鼓励性套话，不要编造数据里没有的结论。"
)

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)


def _clean_narrative(raw: str, patterns: list[str]) -> str:
    text = _THINK_BLOCK.sub("", raw or "")
    text = _UNCLOSED_THINK.sub("", text).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    if text:
        return text
    if patterns:
        return "；".join(pattern.rstrip("。") for pattern in patterns[:3]) + "。"
    return "已根据你的实际作答生成学习建议，请按推荐顺序开始学习。"


class ItemRejected(Exception):
    pass


_UNITS = (r"(?:毫米每秒|毫米|厘米|米|秒|分钟|小时|天|周|个?月|年|度|层|次|"
          r"个|倍|%|％|千克|公斤|牛|安|伏)?")
# 数值部分允许多字中文数字（一万、二十五），第一版只认单字，
# 「一万小时」这种最常见的写法直接漏掉了。
_NUMERIC_OPT = re.compile(r"^\s*(?:[零〇一二两三四五六七八九十百千万亿]+|[\d\.]+)\s*"
                          + _UNITS + r"\s*$")


def _is_numeric(text: str) -> bool:
    """判断一个选项是不是纯数值型（可带单位）。

    数值型和文字型选项的验证方式不同，见 vet() 关 1 的注释。
    """
    t = (text or "").strip()
    return bool(t) and bool(_NUMERIC_OPT.match(t))


# 禁止性表述。切片里带这些词的分句，说的是"不该怎么做"。
_PROHIBIT = re.compile(r"切勿|禁止|严禁|不得|不应|不能|不可|避免|防止|否则|不要")


def _forbidden_by(option: str, body: str, floor: float = 0.30) -> bool:
    """判断这个选项描述的是不是切片明确禁止的做法。是的话它是合格干扰项。

    这条规则是红队测出来的，而且是命题环节最容易出错的地方。

    背景：干扰项的审核逻辑是"覆盖率过高 → 可能同样成立 → 拒收"。
    但在安全和操作规程类内容里，**最好的干扰项恰恰是手册明令禁止的做法**。
    比如 KB-017 写「切勿在未按住解除按钮时强行驱动」，
    那么「直接强行反向驱动」就是一个再标准不过的干扰项 ——
    可它跟切片的二元组覆盖率高达 0.60，直接被判成"可能也对"毙掉了。

    二元组重合看不见否定：「强行驱动」和「切勿强行驱动」在词面上几乎一样。
    照这么卡下去，模型只能去造那些一眼假的干扰项（「更换伺服电机」），
    题目的区分度就废了 —— 学习者不用会也能排除。

    做法：把切片按句切开，只在带禁止词的分句里比对。选项和某个"禁止句"
    高度重合，说明它讲的正是被禁止的做法，放行。

    反向保护：如果**选项自己**也带禁止词，那它是在复述规则、本身成立，
    不能享受这个豁免，否则会把正确说法当成干扰项放进去。
    """
    if _PROHIBIT.search(option or ""):
        return False
    hi_forbid, hi_plain = 0.0, 0.0
    # 按逗号也切开。禁止句常和正确做法挤在同一个句号里：
    #   「加注时必须打开排脂口，禁止在封闭状态下加注，否则会顶坏油封」
    # 只按句号切，整句都会被当成禁止句，连「打开排脂口后加注」这个
    # 正确做法都会被误判成"被禁止的做法"，等于把正确答案放进干扰项。
    # 禁止的作用域是分句级的，切分粒度必须跟上。
    for clause in re.split(r"[。；！\n，,、]", body or ""):
        if not clause.strip():
            continue
        r = overlap_ratio(option, clause)
        if _PROHIBIT.search(clause):
            hi_forbid = max(hi_forbid, r)
        else:
            hi_plain = max(hi_plain, r)
    # 用比较而不是绝对阈值：判据是"这个选项更像禁止句还是更像正文"。
    # 绝对阈值定不准 —— 干扰项通常只借用禁止句的一部分措辞，
    # 「直接强行反向驱动」对「切勿在未按住解除按钮时强行驱动」只有 0.40，
    # 卡 0.5 会漏，卡 0.3 又会把正文里的普通句子误判成禁止句。
    # 比较式天然免疫这个标定问题。
    return hi_forbid >= floor and hi_forbid > hi_plain


class ExaminerAgent:
    name = "命题Agent"

    def __init__(self, llm, retriever: Retriever, kp_index: dict):
        self.llm = llm
        self.retriever = retriever
        self.kp_index = kp_index
        self.auditor = AuditAgent(llm, retriever)
        self.generated: dict[str, dict] = {}
        self.rejects: list[dict] = []
        self.no_output = 0        # 模型没给出可解析的题
        self.llm_errors = 0       # 模型调用失败（限流/额度耗尽/网络）
        self.last_error = ""
        self.requests = 0         # 命题请求次数（一次 make_item 算一次）

    def _auditor_rejects_distractor(self, option: str, source_id: str) -> bool:
        """Only certify a distractor with an explicit semantic rejection."""
        try:
            verdict = self.auditor.semantic_verdict(
                Claim(text=option, source_id=source_id)
            )
        except Exception:  # noqa: BLE001 - verifier failure must fail closed
            return False
        return verdict in (VERDICT_UNSUPPORTED, VERDICT_CONTRADICTED)

    # ---- 一、读自述做分析 ----

    def analyze(self, background: dict, text: str) -> dict:
        """读自述，给出起点判断。模型不可用时退回规则。

        这一步的产出只影响测评的起点和顺序，不影响最终掌握度 ——
        掌握度始终由 BKT 从实际作答算出。分析错了会多花两道题纠正，
        不会得出错误结论。
        """
        rule = self._rule_analyze(background)
        raw = ""
        try:
            raw = self.llm.run(
                task="analyze_intake", system=_ANALYZE_SYS,
                user=json.dumps({"自述": text, "已抽取背景": background},
                                ensure_ascii=False),
                context={"background": background, "text": text},
                json_mode=True,
            )
        except Exception:                                   # noqa: BLE001
            raw = ""
        data = parse_json(raw) if raw else {}

        out = dict(rule)
        lvl = data.get("entry_level")
        if isinstance(lvl, int) and config.DIFFICULTY_MIN <= lvl <= config.DIFFICULTY_MAX:
            out["entry_level"] = lvl
        if isinstance(data.get("summary"), str) and data["summary"].strip():
            out["summary"] = data["summary"].strip()[:160]
        # 重点方向必须是真实存在的知识点，模型编出来的名字一律丢弃
        names = {v["name"]: k for k, v in self.kp_index.items()}
        focus = [names[n] for n in (data.get("focus") or [])
                 if isinstance(n, str) and n in names]
        if focus:
            out["focus"] = focus[:4]
        if isinstance(data.get("caution"), str):
            out["caution"] = data["caution"].strip()[:100]
        return out

    def _rule_analyze(self, bg: dict) -> dict:
        edu = str(bg.get("education", ""))
        hours = float(bg.get("hands_on_hours", 0) or 0)
        lvl = 1
        if edu in ("博士", "硕士"):
            lvl += 1
        elif edu == "本科":
            lvl += 1
        if hours >= 40:
            lvl += 1
        if hours >= 300:
            lvl += 1
        lvl = max(config.DIFFICULTY_MIN, min(config.DIFFICULTY_MAX, lvl))
        # 有实操没系统学过 → 先看原理；学历高但没上过手 → 先看安全和操作
        focus = ["KP-13", "KP-02"] if hours < 40 else ["KP-03", "KP-10"]
        return {
            "summary": f"{edu or '背景未明'}，累计实操约 {int(hours)} 小时。",
            "entry_level": lvl, "focus": focus,
            "caution": "自述为主观描述，起点判断仅用于安排测评顺序。",
            "source": "rule",
        }

    # ---- 二、按需命题 ----

    def make_item(self, kp: str, difficulty: int, avoid: set[str] | None = None,
                  tries: int = 3) -> dict | None:
        """为某个知识点按指定难度出一道题。过不了审就返回 None。"""
        avoid = avoid or set()
        self.requests += 1
        chunks = self.retriever.by_kp(kp)
        if not chunks:
            return None
        node = self.kp_index[kp]

        for attempt in range(tries):
            chunk = chunks[attempt % len(chunks)]
            # 模型侧的任何故障都在这里吃掉，返回 None 让调用方回退题库。
            #
            # 这一条是被免费额度逼出来的：额度耗尽、限流打满、网络断开时，
            # RealLLM 会抛 LLMError。原来没有捕获，异常一路冒到编排层，
            # **整条链路直接崩**。而这个场景下正确的行为是降级 ——
            # 命题本来就是增强项，没有它系统照样能用固定题库跑完。
            # 演示现场额度用完还能继续演，和当场白屏，是两回事。
            try:
                raw = self.llm.run(
                    task="make_item", system=_ITEM_SYS,
                    user=json.dumps({
                        "知识点": node["name"], "目标难度": difficulty,
                        "资料": {"id": chunk.id, "标题": chunk.title,
                                 "正文": chunk.text},
                        "已有题干": sorted(avoid)[:5],
                    }, ensure_ascii=False),
                    context={"chunk": {"id": chunk.id, "title": chunk.title,
                                       "text": chunk.text},
                             "kp": kp, "kp_name": node["name"],
                             "difficulty": difficulty,
                             "seed": f"{kp}:{difficulty}:{attempt}"},
                    json_mode=True,
                )
            except Exception as exc:                        # noqa: BLE001
                self.llm_errors += 1
                self.last_error = str(exc)[:120]
                return None
            data = parse_json(raw)
            if not data:
                # 与"过审失败"分开计数。两者都会导致回退题库，但含义完全不同：
                # 无产出是模型能力或提示词问题，被拒是审核在起作用。
                # 混在一个通过率里报出来，等于把闸门的功劳和模型的短板搅成一团，
                # 答辩时说不清"你们拒了多少道"。
                self.no_output += 1
                continue
            item = {
                "id": f"G-{kp}-{difficulty}-{attempt}",
                "kp": kp, "level": difficulty,
                # 难度是**请求的**，不是实测的。生成题没有作答数据，
                # 无法像题库题那样标定区分度。这个标记必须一路带到界面和报告里 ——
                # 难度会喂给自适应选题，拿一个未经标定的数当真，
                # 等于用没校准的尺子去挑下一道题。
                "level_source": "requested",
                "stem": str(data.get("stem", "")).strip(),
                "options": [str(o).strip() for o in (data.get("options") or [])],
                "answer": data.get("answer"),
                "source_id": chunk.id,
                "explain": str(data.get("explain", "")).strip(),
                "origin": "generated",
            }
            try:
                self.vet(item)
            except ItemRejected as exc:
                self.rejects.append({"item": item, "why": str(exc)})
                continue
            if item["stem"] in avoid:
                continue
            self.generated[item["id"]] = item
            return item
        return None

    # ---- 三、命题审核 ----

    def vet(self, item: dict) -> None:
        """四关审核。任何一关不过就抛 ItemRejected。

        这是整个命题环节的闸门。放松这里等于让系统拿错尺子量人。
        """
        opts = item.get("options") or []
        if len(opts) < 3:
            raise ItemRejected(f"选项只有 {len(opts)} 个")
        if not isinstance(item.get("answer"), int) or not 0 <= item["answer"] < len(opts):
            raise ItemRejected("正确答案序号越界或缺失")
        if len(item.get("stem", "")) < 6:
            raise ItemRejected("题干过短")
        if len({o for o in opts if o}) != len(opts):
            raise ItemRejected("选项存在重复或空值")

        chunk = self.retriever.get(item.get("source_id") or "")
        if chunk is None:
            raise ItemRejected(f"引用切片 {item.get('source_id')} 不存在")
        body = f"{chunk.title} {chunk.text}"
        pool = numbers_in(body)

        right = opts[item["answer"]]

        # 关 1：正确答案必须被切片支撑。
        # 数值型选项和文字型选项要分开判 —— 这是第一版栽的跟头：
        # 「0.5」这种纯数字选项，分词后是 ["0","5"]，只要正文里任意位置
        # 出现过 0 和 5，文本覆盖率就是 1.00，判据完全失效。
        # 数值型的正确判据是"这个数在不在知识库里"，黑白分明，不该绕道文本相似度。
        if _is_numeric(right):
            miss = numbers_in(right) - pool
            if miss:
                raise ItemRejected(f"正确答案的数值 {sorted(miss)} 在切片中不存在")
        else:
            r_right = overlap_ratio(right, body)
            if r_right < config.ITEM_ANSWER_MIN:
                raise ItemRejected(
                    f"正确答案与切片的证据覆盖率仅 {r_right:.2f}，"
                    f"低于 {config.ITEM_ANSWER_MIN}")

        # 关 2：每个干扰项都必须**不**成立。
        # 一道题有两个正确答案，比答案错了还糟：学习者选了另一个对的会被判错，
        # 掌握度往反方向走，而且没有任何环节会发现。
        for i, o in enumerate(opts):
            if i == item["answer"]:
                continue
            if _is_numeric(o):
                # 数值干扰项必须至少有一个数不在知识库里，否则它可能也对
                if numbers_in(o) and not (numbers_in(o) - pool):
                    raise ItemRejected(
                        f"干扰项「{o}」的数值全部出现在切片中，可能同样成立")
            else:
                r = overlap_ratio(o, body)
                if (r >= config.ITEM_DISTRACTOR_MAX
                        and not _forbidden_by(o, body)
                        and not self._auditor_rejects_distractor(o, chunk.id)):
                    raise ItemRejected(
                        f"干扰项「{o[:18]}」的覆盖率 {r:.2f} 过高，可能同样成立")

        # 关 3：题干不得引入切片以外的数值。
        # 只查题干，**不查选项** —— 干扰项按定义就该是知识库里没有的数，
        # 拿这条去卡选项等于禁止出错误选项，第一版就是这么把所有题都毙掉的。
        extra = numbers_in(item["stem"]) - pool
        if extra:
            raise ItemRejected(f"题干出现切片中没有的数值 {sorted(extra)}")

        # 关 4：长度线索。最长的那个恰好是答案，是最常见的出题瑕疵，
        # 学习者不用会也能蒙对，这道题的信息量就废了。
        #
        # 只对文字型选项生效。纯数值选项的长度反映的是量级不是正确性
        # ——「10000」比「2」长，但没有哪个学习者会因此认为它是答案。
        # 第一版没区分，把「一万小时 / 三年」这类完全合格的题全毙了。
        if not all(_is_numeric(o) for o in opts):
            lens = [len(o) for o in opts]
            if (lens[item["answer"]] == max(lens) and min(lens) > 0
                    and max(lens) >= min(lens) * 2):
                raise ItemRejected("正确答案显著长于全部干扰项，存在长度线索")

        # 关 5：结构质量。
        #
        # 前四关问的是"这道题对不对"，这一关问的是"这道题好不好用"。
        # 两者不能互相替代：一道题可以完全正确、出处清楚、干扰项都不成立，
        # 却仍然测不出任何东西 —— 干扰项一眼假、两个干扰项几乎一样、
        # 带「以上都对」这类兜底项。
        #
        # 这些不是错题，前四关放它们过去是对的。但拿它们估掌握度，
        # BKT 的蒙对率 p_G 就不再是 0.25，估出来的掌握概率整体偏高，
        # 而且偏多少不知道。**题不好，尺子就不准。**
        #
        # 只拦 block 级；warn 级记进 quality_warns，交付出去但留痕，
        # 让报告能统计"生成题里有多少道带瑕疵"。
        q = structural(item, body)
        if not q.usable:
            raise ItemRejected("；".join(f.detail for f in q.flaws
                                         if f.severity == "block"))
        item["quality_score"] = q.score
        item["quality_warns"] = [f.code for f in q.flaws if f.severity != "info"]

    # ---- 四、综合分析 ----

    def synthesize(self, diagnosis, log: list[dict]) -> dict:
        """把算好的数据交给模型写一段客观描述。

        **数字一律不交给模型算。** 模型只做归纳表述，算出来的掌握度、
        盲区、可信度全部由规则产出后传进去。这样报告读起来是自然语言，
        但每一个数都可复算 —— 被追问时能当场翻出来源。
        """
        patterns = self._patterns(diagnosis, log)
        raw = ""
        try:
            raw = self.llm.run(
                task="synthesize", system=_SYNTH_SYS,
                user=json.dumps({
                    "整体掌握概率": diagnosis.overall,
                    "盲区知识点": [m.name for m in diagnosis.mastery
                                   if m.kp in diagnosis.gaps][:6],
                    "掌握牢固": [m.name for m in diagnosis.mastery
                                 if m.status == "strong"][:4],
                    "作答模式": patterns,
                }, ensure_ascii=False),
                context={"patterns": patterns, "diagnosis": diagnosis},
            )
        except Exception:                                   # noqa: BLE001
            raw = ""
        return {"narrative": _clean_narrative(raw, patterns), "patterns": patterns}

    def _patterns(self, diagnosis, log: list[dict]) -> list[str]:
        """规则统计出的作答模式。这些是给模型的原料，也是可复算的证据。"""
        out = []
        if not log:
            return out

        # 难度断层：低难度稳、高难度崩，说明是应用能力而非记忆问题
        easy = [s for s in log if s.get("level", 3) <= 2]
        hard = [s for s in log if s.get("level", 3) >= 4]
        if len(easy) >= 2 and len(hard) >= 2:
            ea = sum(s["correct"] for s in easy) / len(easy)
            ha = sum(s["correct"] for s in hard) / len(hard)
            if ea - ha >= 0.4:
                out.append(f"基础题正确率 {ea:.0%}、进阶题 {ha:.0%}，"
                           "断层明显，属于记忆到应用的转化没完成")

        # 安全类专项：这一类错了后果最重，单独拎出来
        safety = [s for s in log
                  if "安全" in (self.kp_index.get(s["kp"], {}).get("tags") or [])]
        if len(safety) >= 2:
            sa = sum(s["correct"] for s in safety) / len(safety)
            if sa < 0.6:
                out.append(f"安全规程相关题目正确率仅 {sa:.0%}，"
                           "上岗前须优先补齐")

        # 追问命中：追问是为消歧而加的，结果值得单独说
        probes = [s for s in log if s.get("probe")]
        if probes:
            out.append(f"触发 {len(probes)} 次动态追问，"
                       "用于分辨失误与真实盲区")

        # 估计不稳：短测评的固有局限，必须说出来
        low = [m for m in diagnosis.mastery if 0 < m.asked and m.confidence < 0.75]
        if low:
            out.append(f"{len(low)} 个知识点的作答样本不足，"
                       "结论可信度有限，建议补测")
        return out
