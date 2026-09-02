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
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock

from core.model_router import (
    AUTH,
    INVALID_RESPONSE,
    MODEL_UNAVAILABLE,
    NETWORK,
    PROVIDER,
    RATE_LIMIT,
    REQUEST,
    SmartModelRouter,
    build_default_specs,
)

_SENT = re.compile(r"[^。！？；\n]+[。！？；]?")


def _load_project_env(env_path: str | Path | None = None) -> None:
    """Load local AgentEdu settings without overriding the process environment."""
    env_path = (Path(env_path) if env_path is not None
                else Path(__file__).resolve().parents[1] / ".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key.startswith("AGENTEDU_"):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            os.environ.setdefault(key, value)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text) if len(s.strip()) >= 8]


class LLMError(RuntimeError):
    pass


@dataclass
class ModelCallError(Exception):
    kind: str
    status: int | None
    summary: str
    latency_ms: int
    code: str = ""

    def __str__(self) -> str:
        return self.summary


@dataclass(frozen=True)
class ModelResult:
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


class _InFlightCall:
    """One cache-key owner and the callers waiting for its terminal outcome."""

    def __init__(self) -> None:
        self.done = Event()
        self.error: LLMError | None = None


_INVALID_RESPONSE_SUMMARY = "响应结构或用量无效"
_DEFAULT_MODEL_ID = "MiniMax-M3"
_STRONG_MODEL_ID = "deepseek-v4-pro"


def classify_http_error(status: int, body: str) -> str:
    lower = body.lower()
    unavailable_markers = (
        "model_not_found",
        "model does not exist",
        "model not found",
        "access denied for model",
    )
    if status == 401:
        return AUTH
    if status in (403, 404) or (
            status == 400 and any(marker in lower for marker in unavailable_markers)):
        return MODEL_UNAVAILABLE
    if status == 429:
        return RATE_LIMIT
    if 500 <= status < 600:
        return PROVIDER
    return REQUEST


class OpenAIAdapter:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def post(self, payload: dict, timeout: int) -> tuple[dict, int]:
        started = time.perf_counter()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "ignore")[:300]
            except Exception:
                body = ""
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            kind = classify_http_error(exc.code, body)
            code = (
                "response_format_unsupported"
                if exc.code == 400 and "response_format" in body.lower()
                else f"http_{exc.code}"
            )
            raise ModelCallError(
                kind,
                exc.code,
                f"{kind}:{code}",
                elapsed_ms,
                code,
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            raise ModelCallError(
                INVALID_RESPONSE,
                None,
                "响应不是合法 JSON",
                elapsed_ms,
                "malformed_json",
            ) from exc
        except http.client.HTTPException as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            raise ModelCallError(
                NETWORK,
                None,
                "network:protocol_error",
                elapsed_ms,
                "protocol_error",
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            raise ModelCallError(
                NETWORK,
                None,
                "network:transport_error",
                elapsed_ms,
                "transport_error",
            ) from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return data, elapsed_ms


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
                 rpm: int = 0, budget_calls: int = 0, cache: bool = True, *,
                 router: SmartModelRouter | None = None,
                 adapter: OpenAIAdapter | None = None,
                 adapters: dict[str, OpenAIAdapter] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        strong_model = (models or {"strong": _STRONG_MODEL_ID}).get(
            "strong", "").strip()
        self.models = {"strong": strong_model}
        self.timeout = timeout
        self.retries = retries
        self.price_in = price_in
        self.price_out = price_out
        specs = build_default_specs(
            model, strong_model, timeout, price_in, price_out)
        self.router = router or SmartModelRouter(specs)
        default_adapter = adapter or OpenAIAdapter(self.base_url, api_key)
        self.adapters = (
            dict(adapters)
            if adapters is not None
            else {model_id: default_adapter for model_id in self.router.specs}
        )
        if set(self.adapters) != set(self.router.specs):
            raise ValueError("每个路由模型都必须配置一个请求适配器")
        self.adapter = self.adapters[model]
        self._json_mode_ok = {
            spec.model_id: spec.supports_json_mode
            for spec in self.router.specs.values()
        }
        self._state_lock = RLock()
        self._inflight: dict[str, _InFlightCall] = {}
        self.calls = 0
        self.http_attempts = 0
        self.failures = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.by_task: dict[str, dict] = {}
        self.by_model: dict[str, dict] = {}
        self.fallbacks = 0

        # ---- 免费档位的三条现实约束 ----
        #
        # rpm：每分钟请求数上限。免费额度基本都限 RPM（有的低到个位数），
        # 超了返回 429。靠重试硬扛的话，一次测评几十次调用会把大部分时间
        # 花在退避等待上，演示现场看着就像卡死了。主动限速比被动挨罚好。
        self.rpm = rpm
        self._min_gap = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0
        self._next_attempt_at = 0.0

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
        candidates = self.router.ordered_candidates(task)
        return candidates[0].model_id if candidates else self.model

    def cost(self) -> float:
        """按配置单价折算累计成本，单位元。

        保留 6 位小数而不是 4 位：单次调用的成本在 1e-4 量级，
        4 位小数会把一整次调用舍进误差里，看上去像"没花钱"。
        一次完整测评是几十次调用，这个分辨率不够用。
        """
        return round(self.tokens_in / 1e6 * self.price_in
                     + self.tokens_out / 1e6 * self.price_out, 6)

    def model_status(self) -> dict:
        snap = self.router.snapshot()
        return {
            "mode": "real",
            "strategy": snap["strategy"],
            "models": snap["models"],
            "router": {
                "fallbacks": snap["fallbacks"],
                "all_models_failed": snap["all_models_failed"],
            },
        }

    def stats(self) -> dict:
        status = self.model_status()
        return {
            "calls": self.calls,
            "http_attempts": self.http_attempts,
            "failures": self.failures,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_cny": self.cost(),
            "by_task": self.by_task,
            "by_model": {
                item["id"]: {
                    "calls": item["successes"],
                    "in": item["tokens_in"],
                    "out": item["tokens_out"],
                }
                for item in status["models"]
            },
            "fallbacks": status["router"]["fallbacks"],
            "cache_hits": self.cache_hits,
            "rpm_limit": self.rpm,
            "budget_calls": self.budget_calls,
            "budget_hit": self.budget_hit,
            "json_mode": all(self._json_mode_ok.values()),
            "router": status["router"],
            "models": status["models"],
        }

    def _throttle(self, wait: float) -> None:
        if wait > 0:
            time.sleep(wait)

    def _reserve_http_attempt(self, spec) -> float | None:
        """Reserve budget, circuit probe, and RPM slot before lock-free I/O."""
        with self._state_lock:
            if self.budget_calls and self.http_attempts >= self.budget_calls:
                self.budget_hit = True
                raise LLMError(
                    f"已达调用上限 {self.budget_calls} 次（AGENTEDU_BUDGET_CALLS）")
            if not self.router.begin_attempt(spec.model_id):
                return None
            self.http_attempts += 1
            now = time.monotonic()
            scheduled = max(now, self._next_attempt_at)
            self._next_attempt_at = scheduled + self._min_gap
            self._last_call = scheduled
            return max(0.0, scheduled - now)

    @staticmethod
    def _key(task: str, model: str, system: str, user: str, temp: float,
             json_mode: bool) -> str:
        raw = f"{task}\x00{model}\x00{temp}\x00{int(json_mode)}\x00{system}\x00{user}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _payload(self, spec, system: str, user: str, temp: float,
                 json_mode: bool) -> dict:
        user_text = user
        payload = {
            "model": spec.model_id,
            "temperature": temp,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
        }
        if json_mode and self._json_mode_ok[spec.model_id]:
            payload["response_format"] = {"type": "json_object"}
        elif json_mode:
            payload["messages"][1]["content"] = (
                user_text + "\n请只输出合法 JSON，不要输出 Markdown 代码围栏或额外说明。")
        return payload

    @staticmethod
    def _parse_result(spec, data: dict, latency_ms: int) -> ModelResult:
        try:
            if not isinstance(data, dict):
                raise TypeError("top-level response is not an object")
            usage = data.get("usage", {})
            if not isinstance(usage, dict):
                raise TypeError("usage is not an object")
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise TypeError("content is not text")
            tokens_in = int(usage.get("prompt_tokens", 0))
            tokens_out = int(usage.get("completion_tokens", 0))
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelCallError(
                INVALID_RESPONSE,
                None,
                _INVALID_RESPONSE_SUMMARY,
                latency_ms,
            ) from exc
        return ModelResult(content, tokens_in, tokens_out, latency_ms)

    @staticmethod
    def _is_response_format_error(error: ModelCallError) -> bool:
        return (
            error.kind == REQUEST
            and error.status == 400
            and error.code == "response_format_unsupported"
        )

    def _record_task_usage(self, task: str, tokens_in: int,
                           tokens_out: int) -> None:
        task_rec = self.by_task.setdefault(task, {"calls": 0, "in": 0, "out": 0})
        task_rec["calls"] += 1
        task_rec["in"] += tokens_in
        task_rec["out"] += tokens_out

    def _cache_model_for(self, task: str) -> str:
        role = "strong" if task == "make_item" else "default"
        return next(spec.model_id for spec in self.router.specs.values()
                    if spec.role == role)

    def _finish_inflight(self, cache_key: str, inflight: _InFlightCall,
                         content: str | None = None,
                         error: LLMError | None = None) -> None:
        with self._state_lock:
            if error is None:
                self._cache[cache_key] = content or ""
            else:
                inflight.error = error
            inflight.done.set()
            if self._inflight.get(cache_key) is inflight:
                del self._inflight[cache_key]

    def _wait_for_inflight(self, cache_key: str, inflight: _InFlightCall) -> str:
        inflight.done.wait()
        with self._state_lock:
            if inflight.error is not None:
                raise inflight.error
            if cache_key in self._cache:
                self.cache_hits += 1
                return self._cache[cache_key]
        raise LLMError("相同请求未生成可复用结果")

    def run(self, task: str, system: str, user: str, context: dict | None = None,
            json_mode: bool = False, temperature: float | None = None) -> str:
        temp = TASK_TEMP.get(task, 0.2) if temperature is None else temperature
        cache_key = self._key(
            task, self._cache_model_for(task), system, user, temp, json_mode)
        inflight: _InFlightCall | None = None
        wait_for: _InFlightCall | None = None
        if self.cache_enabled:
            with self._state_lock:
                if cache_key in self._cache:
                    self.cache_hits += 1
                    return self._cache[cache_key]
                inflight = self._inflight.get(cache_key)
                if inflight is None:
                    inflight = _InFlightCall()
                    self._inflight[cache_key] = inflight
                else:
                    wait_for = inflight
            if wait_for is not None:
                return self._wait_for_inflight(cache_key, wait_for)

        try:
            return self._run_candidates(task, system, user, temp, json_mode,
                                        cache_key, inflight)
        except Exception as exc:
            if inflight is not None:
                waiter_error = (exc if isinstance(exc, LLMError)
                                else LLMError("共享请求执行失败"))
                self._finish_inflight(cache_key, inflight, error=waiter_error)
            raise

    def _run_candidates(self, task: str, system: str, user: str, temp: float,
                        json_mode: bool, cache_key: str,
                        inflight: _InFlightCall | None) -> str:
        candidates = self.router.ordered_candidates(task)
        if not candidates:
            self.router.record_all_models_failed()
            with self._state_lock:
                self.failures += 1
            raise LLMError(f"没有健康模型可执行任务 {task}")

        last_error: ModelCallError | None = None
        previous_model = ""
        for spec in candidates:
            candidate_started = False
            protocol_downgraded = False
            retry_index = 0
            while retry_index < self.retries:
                wait = self._reserve_http_attempt(spec)
                if wait is None:
                    break
                if not candidate_started:
                    if previous_model:
                        self.router.record_fallback(previous_model, spec.model_id)
                        with self._state_lock:
                            self.fallbacks = self.router.fallbacks
                    previous_model = spec.model_id
                    candidate_started = True
                try:
                    self._throttle(wait)
                    payload = self._payload(spec, system, user, temp, json_mode)
                    data, latency_ms = self.adapters[spec.model_id].post(
                        payload, spec.timeout)
                    result = self._parse_result(spec, data, latency_ms)
                except ModelCallError as exc:
                    last_error = exc
                    self.router.record_failure(
                        spec.model_id, exc.kind, exc.status, exc.latency_ms)
                    if (json_mode and not protocol_downgraded
                            and self._is_response_format_error(exc)):
                        with self._state_lock:
                            self._json_mode_ok[spec.model_id] = False
                        self.router.record_json_downgrade(spec.model_id)
                        protocol_downgraded = True
                        continue
                    shared_provider = len({
                        id(provider) for provider in self.adapters.values()
                    }) == 1
                    if exc.kind == REQUEST or (
                            exc.kind == AUTH and shared_provider):
                        with self._state_lock:
                            self.failures += 1
                        raise LLMError(
                            f"调用模型失败（{spec.model_id}）: {exc}") from exc
                    if exc.kind in {AUTH, MODEL_UNAVAILABLE}:
                        break
                    retry_index += 1
                    if retry_index < self.retries:
                        time.sleep(min(8.0, 1.5 ** (retry_index - 1)))
                        continue
                    break
                except BaseException:
                    self.router.release_attempt(spec.model_id)
                    raise
                self.router.record_success(
                    spec.model_id,
                    result.latency_ms,
                    result.tokens_in,
                    result.tokens_out,
                )
                with self._state_lock:
                    self.calls += 1
                    self.tokens_in += result.tokens_in
                    self.tokens_out += result.tokens_out
                    self._record_task_usage(task, result.tokens_in, result.tokens_out)
                    model_rec = self.by_model.setdefault(
                        spec.model_id, {"calls": 0, "in": 0, "out": 0})
                    model_rec["calls"] += 1
                    model_rec["in"] += result.tokens_in
                    model_rec["out"] += result.tokens_out
                if inflight is not None:
                    self._finish_inflight(cache_key, inflight, content=result.content)
                return result.content

        self.router.record_all_models_failed()
        with self._state_lock:
            self.failures += 1
        detail = str(last_error) if last_error else "所有候选均被熔断"
        raise LLMError(f"调用模型失败（任务 {task}）: {detail}")


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
    """按环境变量选后端。任一 key 缺失就退回 Mock，避免半配置启动。

    真模型必填：
      AGENTEDU_MINIMAX_API_KEY / AGENTEDU_MINIMAX_BASE_URL
                          MiniMax 独立凭据和 OpenAI 兼容端点。
      AGENTEDU_DEEPSEEK_API_KEY / AGENTEDU_DEEPSEEK_BASE_URL
                          DeepSeek 独立凭据和 OpenAI 兼容端点。

    选填：
      AGENTEDU_MODEL      默认及降级模型，固定 MiniMax-M3
      AGENTEDU_MODEL_STRONG
                          强任务主模型，固定 deepseek-v4-pro。该模型重试后
                          仍失败时，自动降级到 AGENTEDU_MODEL。
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
    _load_project_env()
    legacy_key = os.environ.get("AGENTEDU_API_KEY", "").strip()
    minimax_key = os.environ.get(
        "AGENTEDU_MINIMAX_API_KEY", legacy_key).strip()
    deepseek_key = os.environ.get(
        "AGENTEDU_DEEPSEEK_API_KEY", legacy_key).strip()
    if not minimax_key or not deepseek_key:
        return MockLLM(
            hallucination_rate=float(os.environ.get("AGENTEDU_INJECT", "0")),
            numeric_drift=float(os.environ.get("AGENTEDU_DRIFT", "0")),
        )
    base_model = os.environ.get("AGENTEDU_MODEL", _DEFAULT_MODEL_ID).strip()
    strong_model = os.environ.get(
        "AGENTEDU_MODEL_STRONG", _STRONG_MODEL_ID).strip()
    if (base_model, strong_model) != (_DEFAULT_MODEL_ID, _STRONG_MODEL_ID):
        raise ValueError(
            "生产模型配置固定为默认 MiniMax-M3、强模型 deepseek-v4-pro")
    legacy_base_url = os.environ.get("AGENTEDU_BASE_URL", "").strip()
    minimax_base_url = os.environ.get(
        "AGENTEDU_MINIMAX_BASE_URL",
        legacy_base_url or "https://api.minimaxi.com/v1",
    ).strip()
    deepseek_base_url = os.environ.get(
        "AGENTEDU_DEEPSEEK_BASE_URL",
        legacy_base_url or "https://api.deepseek.com",
    ).strip()
    minimax_adapter = OpenAIAdapter(minimax_base_url, minimax_key)
    if (minimax_base_url, minimax_key) == (deepseek_base_url, deepseek_key):
        # 兼容旧版统一网关：同一认证域必须共享实例，确保 401 全链路终止。
        deepseek_adapter = minimax_adapter
    else:
        deepseek_adapter = OpenAIAdapter(deepseek_base_url, deepseek_key)
    adapters = {
        base_model: minimax_adapter,
        strong_model: deepseek_adapter,
    }
    return RealLLM(
        base_url=minimax_base_url,
        api_key=minimax_key,
        model=base_model,
        models={
            "strong": strong_model,
        },
        adapters=adapters,
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
