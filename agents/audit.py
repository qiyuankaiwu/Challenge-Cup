"""审核裁判 Agent。

整套方案的地基。逻辑很短，但它是"幻觉率<5%"这个指标能成立的原因：
无依据的断言在输出之前就被拦掉了，不是生成完再去统计有多少条错的。

判据分三层，任何一层不过就打回：
  1. 有没有引用。没有 source_id 直接判无依据，模型不给出处就等于没说。
  2. 引用对不对。断言与所引切片的二元组覆盖率要过线，防止张冠李戴。
  3. 数字准不准。断言里出现的数字必须在切片里出现过。这一条专治
     大模型最典型的失败模式：句式抄对了，把 250 写成 200。

接了真模型以后会多一层蕴含判断，但上面三条规则始终并行跑，取更严的结果。
不要用模型替掉规则，模型自己也会判错，两套独立判据一起用才有意义。
"""

from __future__ import annotations

import json
import re

import config
from core.llm import parse_json
from core.retrieval import Retriever, numbers_in, overlap_ratio, tokenize
from core.schema import (Claim, VERDICT_CONTRADICTED, VERDICT_SUPPORTED,
                         VERDICT_UNSUPPORTED)


_SCOPE_EXPANSION_CUES = (
    "所有", "任何情况下", "任何品牌", "一律", "无论", "都必须", "都使用",
)
_SCOPE_LIMIT_CUES = (
    "为例", "仅适用", "仅在", "因控制器而异", "实际机型", "具体机型",
    "机型手册", "现场规程", "通用数值标准",
)
_RELAXATION_CUES = (
    "无需", "不必", "不需要", "不用", "完全替代", "可以省略",
)
_AUTOMATIC_COMPLETION_CUES = (
    "自动验证", "自动核对", "自动确认", "自动校验",
)
_REQUIREMENT_CUES = (
    "必须验证", "必须核对", "必须确认", "需要验证", "需要核对",
    "需要确认", "应当验证", "应当核对", "应当确认", "仍需验证",
    "仍需核对", "仍需确认", "另行验证", "前提满足",
)
_RISK_REQUIREMENT = re.compile(
    r"(?:错误|不正确)(?:的)?[^，。；]{0,16}(?:时)?可能(?:导致|触发)"
)

_SYSTEM = (
    "你是专业内容的审核员。给你一条陈述和一段资料，判断资料是否支持该陈述。"
    "只输出 JSON：{\"verdict\": \"supported|unsupported|contradicted\", \"reason\": \"简短理由\"}。"
    "资料没提到的内容一律判 unsupported，不要凭常识补充。"
)


def semantic_boundary_issue(claim_text: str, evidence_text: str) -> str | None:
    """识别词面重合无法发现的范围扩大与条件取消。

    这里只处理可由原文边界词直接证明的情况，不把它冒充通用 NLI：开放式
    语义蕴含仍交给真模型复核。两类规则都要求证据中存在相反的限制信号，
    避免仅因断言出现“必须”或“自动”等词就误伤。
    """
    expands_scope = any(cue in claim_text for cue in _SCOPE_EXPANSION_CUES)
    limits_scope = any(cue in evidence_text for cue in _SCOPE_LIMIT_CUES)
    preserves_scope = any(cue in claim_text for cue in _SCOPE_LIMIT_CUES)
    if expands_scope and limits_scope and not preserves_scope:
        return "断言把资料限定的适用范围扩大成了无条件通用结论"

    relaxation_cues = _RELAXATION_CUES + _AUTOMATIC_COMPLETION_CUES
    relaxation = next((cue for cue in relaxation_cues if cue in claim_text), None)
    evidence_relaxes = any(cue in evidence_text for cue in relaxation_cues)
    has_requirement = (
        any(cue in evidence_text for cue in _REQUIREMENT_CUES)
        or bool(_RISK_REQUIREMENT.search(evidence_text))
    )
    if relaxation and not evidence_relaxes and has_requirement:
        return f"断言用“{relaxation}”取消了资料明确保留的条件或步骤"
    return None


class AuditAgent:
    name = "审核裁判Agent"

    def __init__(self, llm, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever

    def _rule_check(self, claim: Claim) -> tuple[str, str, float]:
        if not claim.source_id:
            return VERDICT_UNSUPPORTED, "未给出知识库引用", 0.0
        chunk = self.retriever.get(claim.source_id)
        if chunk is None:
            return VERDICT_UNSUPPORTED, f"引用的切片 {claim.source_id} 不存在", 0.0
        ratio = overlap_ratio(claim.text, chunk.text)
        if config.NUMERIC_STRICT:
            extra = numbers_in(claim.text) - numbers_in(chunk.text)
            if extra:
                return (VERDICT_CONTRADICTED,
                        f"断言中的数值 {sorted(extra)} 在所引切片中不存在", ratio)
        boundary_issue = semantic_boundary_issue(
            claim.text, f"{chunk.title} {chunk.text}"
        )
        if boundary_issue:
            return VERDICT_CONTRADICTED, boundary_issue, ratio
        if ratio < config.EVIDENCE_MIN:
            return VERDICT_UNSUPPORTED, f"与所引切片的证据覆盖率仅 {ratio:.2f}", ratio
        if config.TERM_STRICT:
            # 张冠李戴之一：断言用了知识库里的特征术语，但所引切片没有这个术语。
            # 索引建在 title + text 上，这里也必须用同一份文本，否则会误伤标题里的词。
            missing = self.retriever.distinctive_in(claim.text) - set(
                tokenize(f"{chunk.title} {chunk.text}"))
            if len(missing) >= config.TERM_MISS_TOLERANCE:
                return (VERDICT_CONTRADICTED,
                        f"断言使用的术语 {sorted(missing)} 不属于所引切片", ratio)
        if config.MISATTRIB_MARGIN > 0:
            # 张冠李戴之二：全库里有别的切片明显更能支撑这条断言，
            # 说明模型把 A 的内容挂到了 B 的出处上。典型形态是报警代码对错含义。
            best_id, best = "", 0.0
            for other in self.retriever.chunks:
                r = overlap_ratio(claim.text, f"{other.title} {other.text}")
                if r > best:
                    best_id, best = other.id, r
            if best_id != chunk.id and best - ratio >= config.MISATTRIB_MARGIN:
                return (VERDICT_CONTRADICTED,
                        f"切片 {best_id} 的支撑度 {best:.2f} 明显高于所引 "
                        f"{chunk.id} 的 {ratio:.2f}，疑似引用错位", ratio)
        return VERDICT_SUPPORTED, f"证据覆盖率 {ratio:.2f}", ratio

    def semantic_verdict(self, claim: Claim) -> str | None:
        """返回真模型的独立语义判定；无有效结论时返回 ``None``。"""
        chunk = self.retriever.get(claim.source_id) if claim.source_id else None
        if chunk is None:
            return None
        raw = self.llm.run(
            task="verify",
            system=_SYSTEM,
            user=json.dumps({"陈述": claim.text, "资料": chunk.text}, ensure_ascii=False),
            json_mode=True,
        )
        if not raw:
            return None
        verdict = parse_json(raw).get("verdict")
        return verdict if verdict in (
            VERDICT_SUPPORTED, VERDICT_UNSUPPORTED, VERDICT_CONTRADICTED) else None

    def review(self, claims: list[Claim]) -> tuple[list[Claim], list[Claim]]:
        kept, dropped = [], []
        for claim in claims:
            verdict, note, ratio = self._rule_check(claim)
            claim.evidence_score = round(ratio, 3)
            llm_verdict = (self.semantic_verdict(claim)
                           if verdict == VERDICT_SUPPORTED else None)
            if llm_verdict and llm_verdict != VERDICT_SUPPORTED:
                # 规则放行、模型拦下，取严的一方
                verdict = llm_verdict
                note = f"{note}；模型判定为 {llm_verdict}"
            claim.verdict = verdict
            claim.audit_note = note
            (kept if verdict == VERDICT_SUPPORTED else dropped).append(claim)
        return kept, dropped

    @staticmethod
    def intercept_rate(kept: list[Claim], dropped: list[Claim]) -> float:
        total = len(kept) + len(dropped)
        return round(len(dropped) / total, 4) if total else 0.0
