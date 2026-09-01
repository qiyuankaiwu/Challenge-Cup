"""大模型客户端。

两个后端：
  RealLLM  走 OpenAI 兼容接口，通义千问 / DeepSeek / 智谱都能接，改 base_url 即可。
  MockLLM  离线确定性桩，不联网、不花钱、输出可复现。

MockLLM 不是摆设。它承担三件事：
  1. 单元测试和 CI 不依赖外部服务，跑得快、结果稳定；
  2. 评委现场没网或者 key 过期时，系统照样能完整演示闭环；
  3. 它可以按指定概率注入幻觉，用来验证审核 Agent 真的拦得住，
     这是"幻觉率"指标能拿出证据的关键，不能靠嘴说。

接口只有一个 run()，task 字段用来区分调用场景，Real 后端会据此挑选温度和
是否强制 JSON 输出，Mock 后端据此路由到对应的桩逻辑。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

_SENT = re.compile(r"[^。！？；\n]+[。！？；]?")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text) if len(s.strip()) >= 8]


class LLMError(RuntimeError):
    pass


# 按任务路由模型。贵的模型只用在真正需要判断力的地方。
#
# 这套系统里三类调用的要求完全不同：
#   命题     要严格遵循约束、不能瞎编 → 用强模型，出错代价最高
#   断言起草 中等，且有审核闸兜底     → 中档模型够用
#   自述解析 / 综合诊断 / 简化改写     → 弱模型足够，且调用量最大
# 全用旗舰模型是白烧钱，全用最便宜的会让命题闸门天天拒收。
TASK_TIER = {
    "make_item": "strong",
    "draft_claims": "mid",
    "verify": "mid",
    "quiz": "mid",
    "analyze_intake": "light",
    "synthesize": "light",
    "diagnose_narrative": "light",
    "simplify": "light",
}

# 各任务的采样温度。命题和核验要稳，叙述类可以松一点。
TASK_TEMP = {
    "make_item": 0.15, "verify": 0.0, "draft_claims": 0.2,
    "quiz": 0.2, "analyze_intake": 0.3, "synthesize": 0.4,
    "diagnose_narrative": 0.4, "simplify": 0.5,
}


class RealLLM:
    """OpenAI 兼容的 chat/completions 接口。用标准库发请求，不引入 sdk。

    兼容层要处理三件现实里一定会遇到的事：

    **重试。** 429（限流）和 5xx 是常态，不是异常。一次测评要调几十次，
    没有退避重试的话，中途挂掉整条链路就断了，演示现场尤其难看。

    **JSON 模式各家不一。** `response_format: {"type":"json_object"}` 是
    OpenAI 的约定，兼容接口不一定支持，不支持时通常直接报 400。
    所以第一次被拒之后自动降级：去掉 response_format，改在提示词里要求只输出
    JSON，再试一次。降级过一次就记住，后续不再白试。

    **用量。** 不统计 token 就不知道一次演示花多少钱，也没法在报告里
    写成本。这里累计 usage，`cost()` 按配置的单价折算。
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60,
                 models: dict | None = None, retries: int = 3,
                 price_in: float = 0.0, price_out: float = 0.0,
                 rpm: int = 0, budget_calls: int = 0, cache: bool = True):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        # models 形如 {"strong": "...", "mid": "...", "light": "..."}，缺项回落到 model
        self.models = models or {}
        self.timeout = timeout
        self.retries = retries
        self.price_in = price_in
        self.price_out = price_out
        self.calls = 0
        self.failures = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.by_task: dict[str, dict] = {}
        self._json_mode_ok = True     # 被拒一次后置 False，不再重试

        # ---- 免费档位的三条现实约束 ----
        #
        # rpm：每分钟请求数上限。免费额度基本都限 RPM（有的低到个位数），
        # 超了返回 429。靠重试硬扛的话，一次测评几十次调用会把大部分时间
        # 花在退避等待上，演示现场看着就像卡死了。主动限速比被动挨罚好。
        self.rpm = rpm
        self._min_gap = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0

        # budget_calls：整个进程的调用次数硬上限。免费额度用完之后，
        # 后续请求会全部失败，而系统会一路回退到规则版 —— 结果是"能跑但
        # 悄悄降级了"，最坏的一种故障。设了上限就会在到达前主动停用模型，
        # 并在 stats 里明确标出来。
        self.budget_calls = budget_calls
        self.budget_hit = False

        # cache：相同请求直接复用。本项目的系统提示词是固定的，
        # 评测批次里同一个知识点会被反复起草，命中率相当可观，
        # 对按次计费和按额度限量的场景都是实打实的节省。
        self.cache_enabled = cache
        self._cache: dict[str, str] = {}
        self.cache_hits = 0

    def model_for(self, task: str) -> str:
        return self.models.get(TASK_TIER.get(task, "mid")) or self.model

    def cost(self) -> float:
        """按配置单价折算累计成本，单位元。

        保留 6 位小数而不是 4 位：单次调用的成本在 1e-4 量级，
        4 位小数会把一整次调用舍进误差里，看上去像"没花钱"。
        一次完整测评是几十次调用，这个分辨率不够用。
        """
        return round(self.tokens_in / 1e6 * self.price_in
                     + self.tokens_out / 1e6 * self.price_out, 6)

    def stats(self) -> dict:
        return {"calls": self.calls, "failures": self.failures,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "cost_cny": self.cost(), "by_task": self.by_task,
                "json_mode": self._json_mode_ok,
                "cache_hits": self.cache_hits,
                "rpm_limit": self.rpm,
                "budget_calls": self.budget_calls,
                "budget_hit": self.budget_hit}

    def _throttle(self) -> None:
        if self._min_gap <= 0:
            return
        wait = self._min_gap - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @staticmethod
    def _key(task: str, model: str, system: str, user: str, temp: float) -> str:
        raw = f"{task}\x00{model}\x00{temp}\x00{system}\x00{user}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def run(self, task: str, system: str, user: str, context: dict | None = None,
            json_mode: bool = False, temperature: float | None = None) -> str:
        temp = TASK_TEMP.get(task, 0.2) if temperature is None else temperature
        model = self.model_for(task)

        ck = self._key(task, model, system, user, temp) if self.cache_enabled else ""
        if ck and ck in self._cache:
            self.cache_hits += 1
            return self._cache[ck]

        if self.budget_calls and self.calls >= self.budget_calls:
            # 额度用尽时**明确失败**，而不是继续发请求让它一个个超时。
            # 上层捕获 LLMError 后会回退到规则版，行为是确定的。
            self.budget_hit = True
            raise LLMError(
                f"已达调用上限 {self.budget_calls} 次（AGENTEDU_BUDGET_CALLS），"
                "为保护免费额度已停止调用模型，后续走规则版")

        want_json = json_mode and self._json_mode_ok
        sys_text = system
        if json_mode and not self._json_mode_ok:
            sys_text = system + "\n严格只输出 JSON，不要任何解释文字或代码块标记。"

        payload = {
            "model": model,
            "temperature": temp,
            "messages": [{"role": "system", "content": sys_text},
                         {"role": "user", "content": user}],
        }
        if want_json:
            payload["response_format"] = {"type": "json_object"}

        last = None
        for attempt in range(self.retries):
            try:
                self._throttle()
                data = self._post(payload)
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", "ignore")[:300]
                except Exception:                              # noqa: BLE001
                    pass
                finally:
                    # HTTPError 既是异常也是可读取的响应对象。读取完错误正文后
                    # 必须主动关闭，否则限流或鉴权失败时会遗留 socket 句柄。
                    exc.close()
                # JSON 模式不被支持：降级一次，之后整个会话都不再用
                if exc.code == 400 and want_json and "response_format" in body:
                    self._json_mode_ok = False
                    payload.pop("response_format", None)
                    payload["messages"][0]["content"] = (
                        system + "\n严格只输出 JSON，不要任何解释文字或代码块标记。")
                    want_json = False
                    continue
                last = f"HTTP {exc.code}: {body}"
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(min(8.0, 1.5 ** attempt))       # 指数退避
                    continue
                break                                          # 401/403 等重试无用
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = str(exc)
                time.sleep(min(8.0, 1.5 ** attempt))
                continue

            self.calls += 1
            usage = data.get("usage") or {}
            ti = int(usage.get("prompt_tokens", 0))
            to = int(usage.get("completion_tokens", 0))
            self.tokens_in += ti
            self.tokens_out += to
            rec = self.by_task.setdefault(task, {"calls": 0, "in": 0, "out": 0})
            rec["calls"] += 1
            rec["in"] += ti
            rec["out"] += to
            try:
                content = data["choices"][0]["message"]["content"]
                if ck:
                    self._cache[ck] = content
                return content
            except (KeyError, IndexError):
                last = f"响应结构异常: {str(data)[:200]}"
                break

        self.failures += 1
        raise LLMError(f"调用模型失败（任务 {task}，重试 {self.retries} 次）: {last}")


_DIGITS = re.compile(r"\d+(?:\.\d+)?")

# 量值：数字后面必须跟单位量词，且前面不能贴着字母或连字符。
# 不加这个约束，「报警SRVO-001」的 001、「J1至J3」的 3、「KB-006」的 006
# 都会被当成量值出题，产出「填入正确数值：报警SRVO-____」这种废题。
# 型号里的数字不是量，改了它得到的是另一个概念，不是同一件事的错误答案。
_QUANTITY = re.compile(
    r"(?<![A-Za-z0-9\-])(\d+(?:\.\d+)?)\s*"
    r"(?=毫米每秒|毫米|厘米|米|秒|分钟|小时|天|周|个?月|年|度|层|次|台|根|"
    r"个|倍|%|％|千克|公斤|牛|安|伏)")


def _quantities(text: str) -> list[str]:
    """抽出文中真正表示"量"的数字。"""
    return [m.group(1) for m in _QUANTITY.finditer(text)]


def _drift_number(text: str, salt: int) -> str:
    """把文中最大的那个数字改掉。制造数值冲突用。

    取最大值而不是第一个，是因为第一个往往落在型号里 ——「T1模式」的 1、
    「SRVO-005」的 005。改型号得到的是另一个概念，不是同一件事的两种说法，
    对齐阶段会直接判成不相关，仲裁分支照样走不到。
    量值（250毫米、1.4米、8层）才是真正会被篡改、也真正需要仲裁的东西。
    """
    cands = list(_DIGITS.finditer(text))
    if not cands:
        return text
    m = max(cands, key=lambda x: float(x.group(0)))
    raw = m.group(0)
    try:
        val = float(raw)
    except ValueError:
        return text
    factor = (2.0, 0.5, 1.5, 3.0, 0.25, 4.0, 0.8)[salt % 7]
    new = val * factor
    out = f"{new:.1f}".rstrip("0").rstrip(".") if "." in raw else str(int(new))
    return text[:m.start()] + out + text[m.end():]


def _re_split(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[，。；、\s]+", text or "") if x.strip()]


class MockLLM:
    """离线桩。所有随机性由内容哈希决定，同样输入永远同样输出。"""

    def __init__(self, hallucination_rate: float = 0.0, numeric_drift: float = 0.0):
        self.hallucination_rate = hallucination_rate
        # 数值漂移：只作用于"专家乙"，把某条断言里的数字改掉。
        # 用途单一 —— 制造数值冲突，好让辩论的仲裁分支被真正走到。
        # 不开这个开关，两位专家在桩上几乎从不冲突，仲裁代码就成了死路，
        # 演示和测试都看不到它工作。仅供测试与演示，正式跑分必须设 0。
        self.numeric_drift = numeric_drift
        self.calls = 0

    # 注入用的假事实。都是看着像模像样、实际与知识库冲突的说法，
    # 用来验证审核环节。真实场景里模型编出来的东西就长这样。
    FAKES = [
        "该操作的标准执行时间为12.5秒，超时系统会自动回退。",
        "行业规范要求此项参数必须设定为额定值的1.8倍。",
        "国家标准规定该项检测每72小时执行一次并留存记录。",
        "控制器会在第三次重试失败后自动切换到备用固件分区。",
    ]

    def _pick(self, seed: str, n: int) -> int:
        return int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16) % max(1, n)

    def run(self, task: str, system: str, user: str, context: dict | None = None,
            json_mode: bool = False, temperature: float = 0.2) -> str:
        self.calls += 1
        context = context or {}
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            return ""
        return handler(context)

    def _task_draft_claims(self, ctx: dict) -> str:
        chunks = ctx.get("chunks", [])
        want = int(ctx.get("n", 4))
        claims = []
        for ch in chunks:
            for sent in split_sentences(ch["text"]):
                claims.append({"text": sent, "source_id": ch["id"]})
                if len(claims) >= want:
                    break
            if len(claims) >= want:
                break
        seed = ctx.get("seed", "")
        if self.numeric_drift > 0 and seed.endswith(":wide") and claims:
            bucket = self._pick(seed + "d", 1000) / 1000.0
            if bucket < self.numeric_drift:
                for k, c in enumerate(claims):
                    drifted = _drift_number(c["text"], self._pick(seed + str(k), 7))
                    if drifted != c["text"]:
                        claims[k] = {**c, "text": drifted}
                        break
        if self.hallucination_rate > 0:
            bucket = self._pick(seed, 1000) / 1000.0
            if bucket < self.hallucination_rate:
                fake = self.FAKES[self._pick(seed + "f", len(self.FAKES))]
                # 故意带上一个看似合理的引用，模拟模型"张冠李戴"的典型幻觉
                cite = chunks[0]["id"] if chunks else None
                claims.insert(len(claims) // 2, {"text": fake, "source_id": cite})
        return json.dumps({"claims": claims}, ensure_ascii=False)

    def _task_diagnose_narrative(self, ctx: dict) -> str:
        gaps = ctx.get("gap_names", [])
        strong = ctx.get("strong_names", [])
        bg = ctx.get("background", "")
        parts = [f"该学习者背景为{bg}。"]
        if strong:
            parts.append("已具备基础的是：" + "、".join(strong[:4]) + "。")
        if gaps:
            parts.append("需要优先补齐的是：" + "、".join(gaps[:5]) + "。")
        parts.append("建议按先安全规范、后基本操作、再参数标定与集成调试的顺序推进。")
        return "".join(parts)

    def _task_quiz(self, ctx: dict) -> str:
        claims = ctx.get("claims", [])
        level = int(ctx.get("difficulty", 2))
        items = []
        for i, c in enumerate(claims[:3]):
            items.append({
                "stem": f"关于{ctx.get('kp_name', '本知识点')}，下列说法是否正确：{c['text']}",
                "type": "judge",
                "answer": True,
                "difficulty": max(1, min(5, level + (i - 1))),
                "source_id": c.get("source_id"),
                "explain": f"依据{c.get('source_id')}，该说法与知识库一致。",
            })
        return json.dumps({"items": items}, ensure_ascii=False)

    def _task_analyze_intake(self, ctx: dict) -> str:
        """离线分析。只把已抽取的背景复述成一句话，不做任何推断。"""
        bg = ctx.get("background", {})
        edu = bg.get("education") or "背景未明"
        hours = int(bg.get("hands_on_hours", 0) or 0)
        return json.dumps({
            "summary": f"{edu}，累计实操约 {hours} 小时，"
                       + ("以理论基础为主" if hours < 40 else "有一定现场经验"),
            "caution": "离线模式下未调用大模型，此结论仅复述已抽取的背景。",
        }, ensure_ascii=False)

    def _task_make_item(self, ctx: dict) -> str:
        """离线命题：从切片里挑一个带数值的事实做成四选一。

        做法是把原文数值当正确答案，扰动出三个干扰项。
        这不是"假装有大模型"，而是一种确定性的变式命题策略：
        题目仍然完全来自知识库，答案就是原文，干扰项是原文数值的倍数变换。
        它照样要走 vet() 的四关审核，一关不过就丢。

        局限要说清楚：这种策略只能出数值类题目，出不了概念辨析题。
        接上真模型后覆盖面会宽得多。
        """
        chunk = ctx.get("chunk") or {}
        text = chunk.get("text", "")
        nums = _quantities(text)
        if not nums:
            return ""
        seed = ctx.get("seed", "")
        raw = nums[self._pick(seed, len(nums))]
        # 找出这个数值所在的那句话，用它做题干
        sent = next((s for s in split_sentences(text) if raw in s), text[:60])
        try:
            val = float(raw)
        except ValueError:
            return ""
        factors = [(2.0, 0.5, 4.0), (0.25, 3.0, 1.5), (5.0, 0.2, 2.5)][
            self._pick(seed + "f", 3)]
        fmt = (lambda v: f"{v:.1f}".rstrip("0").rstrip(".")) if "." in raw else (
            lambda v: str(int(v)))
        wrong = [fmt(val * f) for f in factors]
        opts = [raw] + [w for w in wrong if w != raw]
        if len(opts) < 4:
            return ""
        opts = opts[:4]
        idx = self._pick(seed + "i", 4)
        opts[0], opts[idx] = opts[idx], opts[0]
        stem = sent.replace(raw, "____", 1)
        return json.dumps({
            "stem": f"依据规范，填入正确数值：{stem}",
            "options": opts, "answer": idx,
            "explain": f"原文为 {raw}，见 {chunk.get('id', '')}。",
        }, ensure_ascii=False)

    def _task_synthesize(self, ctx: dict) -> str:
        pats = ctx.get("patterns") or []
        if not pats:
            return "本次测评样本有限，结论仅供参考。"
        return "；".join(pats) + "。"

    def _task_factcheck_query(self, ctx: dict) -> str:
        ch = ctx.get("chunk", {})
        import re as _re
        codes = _re.findall(r"[A-Z]{2,}[-\u2013]?\d{2,}", ch.get("text", ""))
        q = " ".join([ch.get("title", "")] + codes[:2]).strip()
        return json.dumps({"queries": [q] if q else []}, ensure_ascii=False)

    def _task_factcheck_judge(self, ctx: dict) -> str:
        """离线桩的判定：**只看资料文本里有没有出现陈述里的关键片段**。

        刻意不做任何"聪明"的判断。桩的职责是让流程跑通并让测试可复现，
        如果桩自己去猜对错，测出来的就不是系统行为而是桩的行为。
        """
        st = ctx.get("statement", "")
        text = (ctx.get("doc", {}) or {}).get("text", "")
        core = [t for t in _re_split(st) if len(t) >= 4][:4]
        hit = sum(1 for t in core if t in text)
        if hit >= 2:
            return json.dumps({"verdict": "support", "reason": "资料包含该陈述的关键片段",
                               "quote": core[0][:28]}, ensure_ascii=False)
        return json.dumps({"verdict": "unknown", "reason": "资料未涉及该陈述"},
                          ensure_ascii=False)

    def _task_simplify(self, ctx: dict) -> str:
        return "换个说法：" + ctx.get("text", "")


def build_llm() -> RealLLM | MockLLM:
    """按环境变量选后端。没配 key 就自动退回 Mock，永远不会因为缺 key 起不来。

    必填：
      AGENTEDU_API_KEY    模型 key。不填即离线跑，系统照常工作。

    选填：
      AGENTEDU_BASE_URL   OpenAI 兼容端点，默认阿里云百炼
      AGENTEDU_MODEL      默认模型
      AGENTEDU_MODEL_STRONG / _MID / _LIGHT
                          按任务分档，见 TASK_TIER。只填一个也行，
                          缺的自动回落到 AGENTEDU_MODEL。
      AGENTEDU_PRICE_IN / _OUT
                          单价（元/百万 token），用于统计成本。不填则成本为 0。
      AGENTEDU_TIMEOUT    单次请求超时秒数，默认 60
      AGENTEDU_RETRIES    最大重试次数，默认 3

    免费额度专用（详见 docs/接入大模型.md）：
      AGENTEDU_RPM        每分钟请求数上限，主动限速。0 为不限，免费档建议填。
      AGENTEDU_BUDGET_CALLS
                          本进程调用次数硬上限，到顶后主动停用模型走规则版，
                          防止额度悄悄耗尽后系统"能跑但已降级"。
      AGENTEDU_CACHE      相同请求复用，默认开启，填 0 关闭

    仅用于测试与演示：
      AGENTEDU_INJECT     离线桩的幻觉注入率
      AGENTEDU_DRIFT      离线桩的数值漂移率
    """
    key = os.environ.get("AGENTEDU_API_KEY", "").strip()
    if not key:
        return MockLLM(
            hallucination_rate=float(os.environ.get("AGENTEDU_INJECT", "0")),
            numeric_drift=float(os.environ.get("AGENTEDU_DRIFT", "0")),
        )
    base_model = os.environ.get("AGENTEDU_MODEL", "qwen3.7-plus")
    return RealLLM(
        base_url=os.environ.get(
            "AGENTEDU_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=key,
        model=base_model,
        models={
            "strong": os.environ.get("AGENTEDU_MODEL_STRONG", ""),
            "mid": os.environ.get("AGENTEDU_MODEL_MID", ""),
            "light": os.environ.get("AGENTEDU_MODEL_LIGHT", ""),
        },
        timeout=int(os.environ.get("AGENTEDU_TIMEOUT", "60")),
        retries=int(os.environ.get("AGENTEDU_RETRIES", "3")),
        price_in=float(os.environ.get("AGENTEDU_PRICE_IN", "0")),
        price_out=float(os.environ.get("AGENTEDU_PRICE_OUT", "0")),
        rpm=int(os.environ.get("AGENTEDU_RPM", "0")),
        budget_calls=int(os.environ.get("AGENTEDU_BUDGET_CALLS", "0")),
        cache=os.environ.get("AGENTEDU_CACHE", "1") != "0",
    )


def parse_json(raw: str, fallback: dict | None = None) -> dict:
    """模型爱包 markdown 代码块，这里统一剥掉再解析。"""
    if not raw:
        return fallback or {}
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
        txt = re.sub(r"\n?```$", "", txt)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(txt[start:end + 1])
            except json.JSONDecodeError:
                pass
    return fallback or {}
