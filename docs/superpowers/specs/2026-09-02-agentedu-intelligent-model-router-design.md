# AgentEdu 双模型智能路由后端设计

日期：2026-09-02
状态：已确认
范围：后端模型调用链，不包含前端改版

## 1. 背景

AgentEdu 当前通过 `core/llm.py` 的 `RealLLM.run()` 调用 OpenAI 兼容接口，
按任务把命题请求交给 `deepseek-v4-pro`，其余请求交给 `MiniMax-M3`。
强模型重试失败后已经可以顺序降级到默认模型，但当前实现仍是固定分支：

- 没有独立的模型注册表和能力描述；
- 路由只看任务档位，不参考模型近期健康、成功率和延迟；
- 没有熔断与半开恢复，故障模型会被每个请求重复探测；
- 指标仅保存在客户端内部，外部无法读取模型健康状态；
- 适配、路由、重试和统计集中在一个类中，继续扩展会使故障语义变得混乱。

参考编程导航《企业级 AI 网关平台项目教程》的公开项目介绍，吸收其中
“统一模型适配器、策略路由、健康检查、自动 Fallback、限流与模型调用统计”
等架构思想。公开页面只提供功能和流程说明，本设计不依赖其付费源码：
<https://www.codefather.cn/course/2029830328322994177>。

## 2. 目标与非目标

### 2.1 目标

1. 系统只注册 `deepseek-v4-pro` 和 `MiniMax-M3` 两个真实模型。
2. 保持所有业务 Agent 继续调用现有 `llm.run()`，不感知路由实现。
3. 根据任务风险、模型能力、实时健康、近期成功率和延迟生成候选顺序。
4. 对可恢复故障执行模型内重试；重试耗尽后自动切换候选模型。
5. 通过熔断器避免持续请求已知故障模型，并允许冷却后自动恢复。
6. 输出不含密钥和敏感请求内容的模型状态与调用指标。
7. 无密钥或全部模型不可用时，保留现有规则版兜底，业务流程不中断。

### 2.2 非目标

- 不建设独立微服务、数据库、Redis、Prometheus 或 Grafana。
- 不实现用户注册、外部 API Key 签发、余额、充值和支付。
- 不实现 SSE、聊天历史、插件系统或 BYOK。
- 不增加第三个模型，也不恢复 Qwen、MID、LIGHT 等旧模型配置。
- 不用后台定时任务主动消耗 Token 做健康探测；真实请求提供被动探测，
  `evalkit.doctor` 承担显式连通检查。

## 3. 总体结构

```text
业务 Agent
   │  llm.run(task, system, user, ...)
   ▼
SmartModelRouter
   ├── ModelRegistry      两个模型的能力和静态配置
   ├── RoutingPolicy      任务适配 + 健康 + 成功率 + 延迟评分
   ├── CircuitBreaker     closed / open / half_open
   ├── OpenAIAdapter[]    每个模型独立端点/密钥，构造并解析请求
   └── RouterMetrics      调用、成功、失败、延迟、Token、Fallback
   │
   ├── deepseek-v4-pro
   └── MiniMax-M3
```

`build_llm()` 在有密钥时返回带智能路由能力的真实客户端；没有密钥时仍返回
`MockLLM`。`RealLLM` 保留为兼容入口，但内部把候选选择交给路由器。

## 4. 组件设计

### 4.1 ModelSpec 与 ModelRegistry

新增不可变 `ModelSpec`，字段包括：

| 字段 | 含义 |
|---|---|
| `model_id` | 对应供应商的模型 ID |
| `role` | `strong` 或 `default` |
| `task_affinity` | 对各任务的静态适配分 |
| `supports_json_mode` | 是否优先发送 `response_format` |
| `timeout` | 单次请求超时 |
| `price_in` / `price_out` | 可选的百万 Token 单价，仅用于统计 |

注册表启动后只有两项：

- `deepseek-v4-pro`：`strong`，命题 `make_item` 的最高静态适配分；
- `MiniMax-M3`：`default`，其他任务最高，并作为 DeepSeek 的降级候选；
- 普通任务在 MiniMax 不可用时可反向使用健康的 DeepSeek，两个模型均失败后才进入规则版。

生产注册表中的 `supports_json_mode` 均为 `false`，JSON 请求直接使用提示词约束。
适配器仍保留 `response_format` 的兼容与降级能力，供后续模型或兼容端点使用。

模型 ID 从现有两个环境变量读取：

```dotenv
AGENTEDU_MINIMAX_API_KEY=<MiniMax key>
AGENTEDU_MINIMAX_BASE_URL=https://api.minimaxi.com/v1  # 国内站；国际站用 api.minimax.io
AGENTEDU_MODEL=MiniMax-M3
AGENTEDU_DEEPSEEK_API_KEY=<DeepSeek key>
AGENTEDU_DEEPSEEK_BASE_URL=https://api.deepseek.com
AGENTEDU_MODEL_STRONG=deepseek-v4-pro
```

若两个变量解析成相同 ID、空 ID 或出现第三个模型，启动体检应给出明确错误，
不能静默形成未知路由表。

### 4.2 OpenAIAdapter

每个模型拥有独立的适配器实例和 Bearer Key。适配器只负责一次模型调用所需的协议工作：

- 构造 `/chat/completions` 请求；
- 添加 Bearer 鉴权；
- 处理 JSON 模式兼容降级；
- 解析文本、usage 和 HTTP 错误；
- 返回结构化 `ModelResult`，不决定下一次调用哪个模型。

错误统一分类为：

| 类型 | 示例 | 路由行为 |
|---|---|---|
| `auth` | HTTP 401 | 独立供应商不重试当前模型但可降级；统一网关共用凭据时全链路终止 |
| `request` | JSON 降级后仍为普通 400 | 终止，避免把同一错误请求重复发给另一模型 |
| `model_unavailable` | 403、404，或 400 且供应商错误码明确为模型不存在/无权限 | 立即尝试下一候选 |
| `rate_limit` | 429 | 当前模型退避重试，耗尽后切换 |
| `provider` | 5xx | 当前模型退避重试，耗尽后切换 |
| `network` | 超时、断网、DNS | 当前模型退避重试，耗尽后切换 |
| `invalid_response` | 200 但缺少有效 content | 记录失败并切换 |

错误对象只保留类型、状态码和经过截断/清洗的摘要，不保存 API Key、完整提示词
或模型返回正文。

### 4.3 RoutingPolicy

路由器先排除：

1. 处于 `open` 且冷却未结束的模型；
2. 不满足请求必要能力的模型；
3. 已在本次调用中失败的模型。

剩余候选使用可解释的分层决策，不开放任意权重配置，避免为两模型系统引入
无法验证的调参面。固定规则按顺序执行：

- `make_item`：DeepSeek 的任务适配分高于 MiniMax；
- 其他任务：MiniMax 的任务适配分高于 DeepSeek；
- 任一模型处于 `open` 时直接排除，`half_open` 只允许单个探测请求；
- 成功率采用最近 20 次调用的滑动窗口；任一模型样本少于 5 次时维持静态顺序；
- 样本充足且首选模型成功率比备选低至少 20 个百分点时，交换顺序；
- 成功率差小于 5 个百分点、且首选平均延迟超过备选 2 倍时，交换顺序；
- 其他情况保持静态任务顺序；成本本阶段只统计，不参与路由决策。

该规则使系统在没有历史数据时保持现有确定性路由，有足够现场数据后才产生
动态选择，避免冷启动时因随机延迟抖动频繁换模型。

### 4.4 CircuitBreaker

每个模型独立维护状态：

```text
closed --连续 3 次可恢复失败--> open
open   --冷却 60 秒-----------> half_open
half_open --1 次成功----------> closed
half_open --1 次失败----------> open
```

- `rate_limit`、`provider`、`network` 和 `invalid_response` 计入连续失败；
- `model_unavailable` 直接打开熔断器；
- `auth` 不改变单模型健康状态；独立供应商可降级，统一网关共用凭据时终止全链路；
- `request` 属于调用方问题，不改变模型健康状态；
- 成功会清零连续失败计数；
- 时间通过可注入时钟获取，测试无需真实等待 60 秒。

进程重启后健康状态清零。当前 MVP 不持久化熔断状态，避免数据库依赖。

### 4.5 重试与 Fallback

一次调用的顺序为：

1. 路由器生成有序候选；
2. 对第一候选执行最多 `AGENTEDU_RETRIES` 次调用；
3. 记录每次结果并更新熔断器；
4. 可恢复失败耗尽后选择下一候选；
5. 候选成功即返回并缓存；
6. 全部候选失败时抛出 `LLMError`，由现有上层逻辑降级到规则版。

Fallback 后的成功结果继续使用“任务 + 首选模型 + 输入”作为缓存键。同一输入再次
出现时直接复用已审计的结果，不反复探测故障模型；熔断器恢复后的新请求再重新评分。

调用次数预算在每次真实 HTTP 请求前检查，而不是只统计成功响应，防止失败重试和
Fallback 绕过 `AGENTEDU_BUDGET_CALLS`。这一点会修正当前只按成功调用计数的口径。

### 4.6 RouterMetrics

内存指标按模型分别维护：

- HTTP 尝试数、成功数、失败数；
- 成功率；
- 输入/输出 Token；
- 平均延迟和最近一次延迟；
- Fallback 流入、流出次数；
- 熔断状态、连续失败数、冷却剩余秒数；
- 最近错误类型和状态码；
- JSON 协议降级次数。

保留现有 `llm.stats()` 字段，并新增稳定的 `models` 和 `router` 子结构，避免前端或
评测代码依赖内部对象。

## 5. 状态接口

在 `server.py` 新增只读接口：

```http
GET /api/model-status
```

响应示例：

```json
{
  "mode": "real",
  "strategy": "task-aware-health-adaptive",
  "models": [
    {
      "id": "deepseek-v4-pro",
      "role": "strong",
      "health": "closed",
      "cooldown_remaining_seconds": 0,
      "attempts": 12,
      "success_rate": 0.92,
      "avg_latency_ms": 1840,
      "fallback_in": 0,
      "fallback_out": 1,
      "last_error": {"type": "rate_limit", "status": 429}
    },
    {
      "id": "MiniMax-M3",
      "role": "default",
      "health": "closed",
      "cooldown_remaining_seconds": 0,
      "attempts": 16,
      "success_rate": 1.0,
      "avg_latency_ms": 620,
      "fallback_in": 1,
      "fallback_out": 0,
      "last_error": null
    }
  ],
  "router": {
    "fallbacks": 1,
    "all_models_failed": 0
  }
}
```

离线模式返回 `mode: "offline"` 和空模型指标。响应禁止包含 API Key、Authorization、
完整 Base URL 查询参数、提示词、学习者输入或模型输出。

## 6. 与现有业务的兼容

- `ExaminerAgent`、`GenerateAgent`、`AuditAgent` 等调用方不改签名。
- `TASK_TIER` 可保留为任务风险元数据，但不再直接等于模型选择结果。
- `MockLLM` 保持完全确定性，单元测试和 CI 不读取真实密钥、不发计费请求。
- 当前 JSON 模式提示词降级继续保留，并改为按模型记录能力结果。
- 当前限速、预算和缓存继续生效；本阶段仍使用进程级保守上限统一约束两家调用。
- 上层固定题库和规则版兜底逻辑不变。

## 7. 文件边界

| 文件 | 责任 |
|---|---|
| `core/model_router.py` | ModelSpec、指标、路由策略、熔断状态机 |
| `core/llm.py` | OpenAIAdapter、兼容入口、`.env` 装配 |
| `server.py` | `/api/model-status` 只读接口 |
| `evalkit/doctor.py` | 两模型主动体检、路由与健康摘要 |
| `tests/test_model_router.py` | 评分、熔断、恢复和候选顺序 |
| `tests/test_llm_client.py` | HTTP 错误分类、重试、Fallback、协议兼容 |
| `tests/test_server.py` 或现有服务测试 | 状态接口与敏感字段检查 |
| `.env.example`、README、接入文档 | 双模型配置和运维说明 |

前端文件不在本阶段范围内；状态接口先为调试、验收和后续监控页面提供数据。

## 8. 测试与验收

实现必须先写失败测试，并覆盖：

1. 冷启动时命题优先 DeepSeek，普通任务优先 MiniMax；
2. 有足够历史样本后，不健康或明显变慢的首选模型被重新排序；
3. 429/5xx/网络错误按模型重试，耗尽后 Fallback；
4. 401 不重试当前模型；独立供应商允许 Fallback，统一网关不重复请求；
5. 普通 400 不 Fallback，JSON 模式不支持时仅做协议降级；
6. 连续三次失败打开熔断，冷却后进入半开，一次成功关闭；
7. half_open 并发只允许一个探测请求；
8. 失败尝试也消耗调用预算；
9. Fallback 结果可缓存，重复输入不重复付费；
10. `/api/model-status` 在真实和离线模式下结构稳定且不泄露敏感信息；
11. `.env` 中只有两个模型 ID，`.env` 继续被 Git 忽略；
12. 全量现有测试继续通过。

真实验收分两层：

- 无密钥环境：全部单元测试和本地假端点集成测试通过；
- 用户在本机 `.env` 分别填入 MiniMax 与 DeepSeek Key 后：运行 `python3 -m evalkit.doctor`，确认两个模型
  均可调用，再通过受控故障测试观察 `/api/model-status` 的 Fallback 和熔断变化。

## 9. 实施顺序

1. 用纯单元测试实现模型注册、指标和熔断状态机；
2. 把 HTTP 单次调用从 `RealLLM` 中抽成适配器；
3. 接入任务感知评分和候选路由；
4. 修正重试、预算、缓存与 Fallback 的组合语义；
5. 增加状态接口和 doctor 输出；
6. 更新配置文档，运行定向测试与全量回归；
7. 用户填入真实 Key 后完成双模型连通和真实降级验收。
