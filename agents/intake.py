"""学情访谈 Agent。

学习者自己打一段话进来，从里面抽出结构化背景，用来给 BKT 的初始掌握概率
定先验，也用来决定测评从哪个难度起步。

两条通路：
  有真模型 → 让模型抽成 JSON，抽不出来或字段不合法就退回规则。
  无模型   → 纯规则抽取（关键词 + 数字 + 单位换算）。

规则通路不是应付了事的兜底。它承担两件正事：
  1. 演示和离线场景下访谈照样能用，不因为没有 key 就退化成填表；
  2. 给模型输出兜底校验 —— 模型抽出"实操学时 99999"这种，规则能拦。

**这里的抽取结果只影响先验，不直接决定诊断结论。**
先验会随着作答证据累积被冲淡（BKT 的性质），所以就算抽错了，
多答几道题就会被数据纠正回来，不会一错错到底。这是有意的设计：
自述是主观的，学习者容易高估或低估自己，不能让它拍板。
"""

from __future__ import annotations

import json
import re

from core.llm import parse_json
from core.retrieval import cn_to_int

_SYSTEM = (
    "你从学习者的自述里抽取结构化背景信息。只输出 JSON，字段："
    '{"education":"博士|硕士|本科|高职|中职|高中|其他","grade":"",'
    '"major":"","hands_on_hours":0,"goal":""}。'
    "hands_on_hours 是累计动手实操小时数，自述里没提就填 0，不要猜。"
    "自述里没有的信息一律留空或填 0，禁止推断。"
)

_EDU = [
    (r"博士", "博士"), (r"硕士|研究生", "硕士"),
    (r"本科|大一|大二|大三|大四|学士", "本科"),
    (r"高职|大专|专科|高专", "高职"), (r"中职|技校|中专", "中职"),
    (r"高中", "高中"),
]
_GRADE = [
    (r"大一|一年级", "一年级"), (r"大二|二年级", "二年级"),
    (r"大三|三年级", "三年级"), (r"大四|四年级", "四年级"),
    (r"在职|工作|上班|产线|车间", "在职"),
]
_MAJOR = [
    (r"机械|机电|机制", "机械类"), (r"电气|自动化|电子", "电气自动化类"),
    (r"计算机|软件|信息", "计算机类"), (r"工业机器人|机器人", "机器人技术"),
]

# 时长表达。按小时归一。
_HOUR_PATTERNS = [
    (r"(\d+(?:\.\d+)?)\s*(?:个)?小时", 1.0),
    (r"(\d+(?:\.\d+)?)\s*天(?!级)", 8.0),
    (r"(\d+(?:\.\d+)?)\s*周(?!级)", 40.0),
    (r"(\d+(?:\.\d+)?)\s*(?:个)?月(?!级)", 160.0),
    (r"(\d+(?:\.\d+)?)\s*年(?!级)", 1600.0),
]
_CN_DURATION_NUM = r"[零〇一二两三四五六七八九十百千万亿]+"
# 「二年级」「三年级」是学制，不是工龄。不排掉的话，一句
# 「高职电气自动化二年级」会被算成两年实操、折合 3200 小时，
# 先验直接顶到上界，测评开局就问偏。
_NOT_DURATION = r"(?!级|段|检)"
_CN_HOUR = [
    (rf"({_CN_DURATION_NUM})\s*年" + _NOT_DURATION, 1600.0),
    (rf"({_CN_DURATION_NUM})\s*(?:个)?月" + _NOT_DURATION, 160.0),
    (rf"({_CN_DURATION_NUM})\s*周" + _NOT_DURATION, 40.0),
    (rf"({_CN_DURATION_NUM})\s*天" + _NOT_DURATION, 8.0),
    (rf"({_CN_DURATION_NUM})\s*(?:个)?小时", 1.0),
]
# 半个单位单独处理；可同时覆盖「半小时」和「一个半小时」。
_CN_HALF_HOUR = [
    (rf"(?:({_CN_DURATION_NUM})\s*)?(?:个)?半\s*年" + _NOT_DURATION, 1600.0),
    (rf"(?:({_CN_DURATION_NUM})\s*)?(?:个)?半\s*(?:个)?月" + _NOT_DURATION, 160.0),
    (rf"(?:({_CN_DURATION_NUM})\s*)?(?:个)?半\s*周" + _NOT_DURATION, 40.0),
    (rf"(?:({_CN_DURATION_NUM})\s*)?(?:个)?半\s*天" + _NOT_DURATION, 8.0),
    (rf"(?:({_CN_DURATION_NUM})\s*)?(?:个)?半\s*(?:个)?小时", 1.0),
]

# 明确表示"没碰过"的说法。这类要压到 0，不能因为提到了"机器人"就给学时。
_ZERO_HINTS = [r"没(?:有)?(?:接触|碰|做|上手|操作)过", r"零基础", r"完全没",
               r"第一次", r"从没", r"没干过", r"不会用"]


def rule_extract(text: str) -> dict:
    """纯规则抽取。不联网、确定性、可测。"""
    t = (text or "").strip()
    out = {"education": "", "grade": "", "major": "", "hands_on_hours": 0,
           "goal": "", "raw": t}
    if not t:
        return out

    for pat, val in _EDU:
        if re.search(pat, t):
            out["education"] = val
            break
    for pat, val in _GRADE:
        if re.search(pat, t):
            out["grade"] = val
            break
    for pat, val in _MAJOR:
        if re.search(pat, t):
            out["major"] = val
            break

    hours = 0.0
    for pat, mul in _HOUR_PATTERNS:
        for m in re.finditer(pat, t):
            hours = max(hours, float(m.group(1)) * mul)
    for pat, mul in _CN_HOUR:
        for m in re.finditer(pat, t):
            hours = max(hours, (cn_to_int(m.group(1)) or 0) * mul)
    for pat, mul in _CN_HALF_HOUR:
        for m in re.finditer(pat, t):
            base = cn_to_int(m.group(1)) if m.group(1) else 0
            hours = max(hours, ((base or 0) + 0.5) * mul)

    if any(re.search(p, t) for p in _ZERO_HINTS):
        hours = 0.0

    # 上限保护：自述里说"干了二十年"折算出三万多小时，对先验没有额外意义，
    # 反而会把先验顶到上界。截断在一个饱和值上。
    capped = min(float(hours), 2000.0)
    out["hands_on_hours"] = int(capped) if capped.is_integer() else capped

    m = re.search(r"(?:想|希望|打算|目标是|准备)([^。；\n]{2,40})", t)
    if m:
        out["goal"] = m.group(1).strip()
    return out


def merge(rule: dict, model: dict | None) -> dict:
    """模型抽取结果与规则结果合并，规则负责校验。

    模型给的值只在"合法且规则没抽到"时采用。宁可少填，不要填错 ——
    先验填错会让测评开局问偏，虽然后面会被数据纠正，但会多花几道题。
    """
    out = dict(rule)
    if not model:
        return out
    edu_ok = {"博士", "硕士", "本科", "高职", "中职", "高中", "其他"}
    if not out["education"] and model.get("education") in edu_ok:
        out["education"] = model["education"]
    for k in ("grade", "major", "goal"):
        if not out[k] and isinstance(model.get(k), str):
            out[k] = model[k][:40]
    if out["hands_on_hours"] == 0:
        h = model.get("hands_on_hours")
        if isinstance(h, (int, float)) and 0 <= h <= 20000:
            out["hands_on_hours"] = int(min(h, 2000))
    return out


def clarify(bg: dict) -> list[str]:
    """自述里缺什么就问什么。模块级函数，方便前端与测试直接调用。

    只问真缺的字段，不做例行盘问 —— 字段齐了就不问，直接开测。
    """
    qs = []
    if not bg.get("education"):
        qs.append("你目前的学历或在读层次是？")
    if not bg.get("hands_on_hours"):
        qs.append("你实际上手操作过工业机器人吗？大概多久？")
    if not bg.get("major") and not bg.get("grade"):
        qs.append("你的专业方向或者现在的岗位是什么？")
    return qs[:2]


class IntakeAgent:
    name = "学情访谈Agent"

    def __init__(self, llm):
        self.llm = llm

    def parse(self, text: str) -> dict:
        rule = rule_extract(text)
        raw = ""
        try:
            raw = self.llm.run(task="intake", system=_SYSTEM, user=text or "",
                               json_mode=True)
        except Exception:                                  # noqa: BLE001
            raw = ""
        return merge(rule, parse_json(raw) if raw else None)

    def clarify(self, bg: dict) -> list[str]:
        return clarify(bg)
