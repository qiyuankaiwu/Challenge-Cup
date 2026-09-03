"""演示服务。

    python3 server.py        然后打开 http://127.0.0.1:8000

用标准库 http.server，不装 FastAPI。骨架阶段的目标是"评委机器上 clone 下来
直接 python3 server.py 就能跑"，少一个依赖少一个现场翻车的可能。
并发和鉴权后面真要上再换 FastAPI，接口路径不变。

会话存在内存里，进程重启就没了。演示够用，别拿去当生产。
"""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import config
from agents.examiner import ExaminerAgent
from agents.intake import IntakeAgent
from core.cat import AdaptiveSession
from core.demo_items import formal_demo_items
from core.demo_sources import (
    publicly_verified_source_ids,
    validate_demo_source_manifest,
)
from core.llm import MockLLM, build_llm
from core.retrieval import Retriever
from orchestrator import Orchestrator, load_profile
from tools.ingest import ingest, quality_gate, write_stage

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
INCOMING_DIR = ROOT / "data" / "incoming"
UPLOAD_STAGE_DIR = ROOT / "data" / "staged" / "uploads"
# 与 tools.ingest 支持的格式保持完全一致。浏览器端也使用这份名单，避免
# 前端显示可上传、后端却无法切片的情况。
ALLOWED_UPLOAD_SUFFIXES = frozenset({".txt", ".md", ".pdf", ".docx",
                                     ".csv", ".tsv", ".xlsx"})
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 28 * 1024 * 1024
SESSIONS: dict[str, tuple[Orchestrator, object]] = {}
INTERVIEWS: dict[str, AdaptiveSession] = {}
_ITEMS = formal_demo_items(
    json.loads(config.PRETEST_PATH.read_text(encoding="utf-8"))["items"]
)
_KPS = json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]
_KP_INDEX = {k["id"]: k for k in _KPS}
_MODEL_CLIENT = None
_MODEL_CLIENT_LOCK = Lock()


def get_model_client():
    global _MODEL_CLIENT
    if _MODEL_CLIENT is None:
        with _MODEL_CLIENT_LOCK:
            if _MODEL_CLIENT is None:
                _MODEL_CLIENT = build_llm()
    return _MODEL_CLIENT


def model_status_payload() -> dict:
    client = get_model_client()
    if isinstance(client, MockLLM):
        return {
            "mode": "offline",
            "strategy": "deterministic-rules",
            "models": [],
            "router": {"fallbacks": 0, "all_models_failed": 0},
        }
    return client.model_status()


class UploadError(ValueError):
    """资料上传不符合受控摄入规范。"""


def _upload_name(filename: object) -> tuple[str, str]:
    """校验原始文件名；不重命名或猜测扩展名，避免绕过格式门禁。"""
    if not isinstance(filename, str):
        raise UploadError("缺少文件名")
    name = filename.strip()
    if not name or len(name) > 120:
        raise UploadError("文件名不能为空，且不得超过 120 个字符")
    if "\x00" in name or "/" in name or "\\" in name or Path(name).name != name:
        raise UploadError("文件名不能包含路径")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = "、".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
        raise UploadError(f"不支持该格式；仅接收 {allowed}")
    return name, suffix


def _upload_source(value: object) -> str:
    """资料来源说明是复核入口，不能只留一个没有上下文的文件名。"""
    if not isinstance(value, str):
        raise UploadError("请填写资料来源、版本或授权说明")
    source = " ".join(value.split())
    if len(source) < 4:
        raise UploadError("资料来源说明至少填写 4 个字符")
    if len(source) > 500:
        raise UploadError("资料来源说明不得超过 500 个字符")
    return source


def _upload_bytes(encoded: object) -> bytes:
    """严格解码 Base64，并在落盘前执行大小与空文件检查。"""
    if not isinstance(encoded, str) or not encoded:
        raise UploadError("没有读取到文件内容")
    # Base64 最多约为原始大小的 4/3；先拦截异常大的 JSON 字段，避免不必要的解码。
    max_encoded = ((MAX_UPLOAD_BYTES + 2) // 3) * 4
    if len(encoded) > max_encoded:
        raise UploadError("文件超过 20 MiB 限制")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UploadError("文件内容编码无效") from exc
    if not content:
        raise UploadError("不接收空文件")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadError("文件超过 20 MiB 限制")
    return content


def stage_material(filename: object, encoded: object, source_description: object,
                   rights_confirmed: object, *, incoming_dir: Path = INCOMING_DIR,
                   staging_root: Path = UPLOAD_STAGE_DIR) -> dict:
    """把原始资料暂存、切片、质检；此函数绝不调用 apply_to_kb。

    资料先保存在独立的 incoming 目录。切片结果进入本次上传独立的暂存目录，
    合格与隔离内容都留下，等待人工查看来源、内容和质量报告后再走正式入库流程。
    """
    original_name, _ = _upload_name(filename)
    source = _upload_source(source_description)
    if rights_confirmed is not True:
        raise UploadError("请确认资料可用于本项目，且不含学习者个人信息")
    content = _upload_bytes(encoded)

    upload_id = uuid.uuid4().hex[:12]
    incoming_dir.mkdir(parents=True, exist_ok=True)
    stored = incoming_dir / f"{upload_id}_{original_name}"
    stored.write_bytes(content)

    # ingest 只生成 sourced / quarantined 的暂存切片；quality_gate 不会删除原文。
    result = ingest([stored])
    if not result["staged"]:
        reasons = "；".join(reason for _, reason in result["skipped"])
        detail = f"（{reasons}）" if reasons else ""
        raise UploadError(
            f"未提取到可复核内容{detail}；请检查文件格式、内容和解析工具"
        )
    quality_gate(result["staged"])
    stage_dir = staging_root / upload_id
    write_stage(result, stage_dir)

    sourced = [s for s in result["staged"] if s.status == "sourced"]
    quarantined = [s for s in result["staged"] if s.status == "quarantined"]
    manifest = {
        "upload_id": upload_id,
        "original_filename": original_name,
        "stored_filename": stored.name,
        "source_description": source,
        "rights_confirmed": True,
        "bytes": len(content),
        "review_state": "pending_manual_review",
        "knowledge_base_written": False,
        "verified": False,
        "next_steps": [
            "人工核对原始资料、文件指纹和每条切片的来源位置",
            "查看隔离原因并处理冲突、错配或待归类内容",
            "完成独立外部证据复核后，才可由人工标记 verified",
        ],
    }
    (stage_dir / "upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "upload_id": upload_id,
        "filename": original_name,
        "bytes": len(content),
        "sourced": len(sourced),
        "quarantined": len(quarantined),
        "unassigned": sum(1 for s in sourced if s.kp is None),
        "skipped": [{"file": name, "reason": reason}
                    for name, reason in result["skipped"]],
        "review_state": manifest["review_state"],
        "knowledge_base_written": False,
        "verified": False,
        "next_steps": manifest["next_steps"],
    }


def _examiner():
    if not config.EXAMINER_ENABLED:
        return None
    return ExaminerAgent(
        get_model_client(),
        Retriever.from_jsonl(config.KB_PATH, demo_only=True),
        _KP_INDEX,
    )


def list_profiles() -> list[dict]:
    out = []
    for p in sorted(config.PROFILE_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        bg = d.get("background", {})
        out.append({
            "id": d["id"], "name": d.get("name", d["id"]),
            "label": f"{bg.get('education', '')}·{bg.get('major', '') or '无专业背景'}",
            "grade": bg.get("grade", ""), "hours": bg.get("hands_on_hours", 0),
            "self_report": bg.get("self_report", ""), "goal": d.get("goal", ""),
        })
    return out


def session_payload(sid: str) -> dict:
    orch, session = SESSIONS[sid]
    kp_index = orch.kp_index
    data = session.to_dict()
    data["session_id"] = sid
    data["state"] = orch.state
    data["kp_index"] = kp_index
    data["path_names"] = [kp_index[k]["name"] for k in session.path]
    # 学习反馈会影响后续难度和资源生成，因此标准答案与解析不能随会话数据
    # 提前下发。提交选择后，由 /api/feedback 在服务端判分并返回本轮解析。
    for resource in data.get("resources", []):
        if resource.get("kind") == "quiz":
            resource["items"] = [public_feedback_item(item)
                                 for item in resource.get("items", [])]
    # 资源展示必须把来源、核实状态和人工备注一并带到前端。只给 source_id
    # 会让评委看不到“这条断言的依据是否已核实”，也无法核对具体来源。
    publicly_verified = publicly_verified_source_ids()
    data["kb"] = {
        c.id: {
            "title": c.title,
            "source": c.source,
            "verified": c.verified and c.id in publicly_verified,
            "source_note": c.source_note,
        }
        for c in orch.retriever.chunks
        if c.demo_eligible
    }
    validate_demo_source_manifest(data["kb"], artifact="在线会话")
    return data


_PUBLIC_FEEDBACK_ITEM_FIELDS = ("stem", "type", "difficulty", "source_id")


def public_feedback_item(item: dict) -> dict:
    """公开反馈题题面，不公开会影响服务端判分的字段。"""
    return {key: item[key] for key in _PUBLIC_FEEDBACK_ITEM_FIELDS if key in item}


def score_feedback_choices(session, kp: str, choices: object
                           ) -> tuple[list[bool], list[dict]]:
    """用会话内标准答案判分，并返回提交后才可公开的逐题结果。"""
    if not isinstance(kp, str) or not kp:
        raise ValueError("缺少要反馈的知识点")
    if not isinstance(choices, list) or not choices:
        raise ValueError("请提交本轮每道题的选择")
    if any(type(choice) is not bool for choice in choices):
        raise ValueError("每道题的选择必须是真正的布尔值")

    resource = next((item for item in session.resources
                     if item.kp == kp and item.kind == "quiz"), None)
    items = resource.items[:4] if resource is not None else []
    if not items:
        raise ValueError("该知识点没有可用于反馈的测试题")
    if len(choices) != len(items):
        raise ValueError(f"本轮应提交 {len(items)} 道题的选择")

    correctness = []
    results = []
    for index, (item, choice) in enumerate(zip(items, choices), 1):
        answer = item.get("answer")
        if type(answer) is not bool:
            raise ValueError(f"第 {index} 题缺少有效的标准答案")
        correct = choice is answer
        correctness.append(correct)
        results.append({
            "index": index,
            "stem": item.get("stem", ""),
            "choice": choice,
            "answer": answer,
            "correct": correct,
            "explain": item.get("explain", ""),
        })
    return correctness, results


_PUBLIC_INTERVIEW_ITEM_FIELDS = (
    "id", "kp", "level", "stem", "options", "level_source", "source_id",
    "origin", "_reason", "_kp_name",
)


def public_interview_item(item: dict | None) -> dict | None:
    """只公开作答所需题目字段；答案始终留在服务端评分。"""
    if item is None:
        return None
    return {key: item[key] for key in _PUBLIC_INTERVIEW_ITEM_FIELDS if key in item}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):   # 静音，演示时控制台不刷屏
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/model-status":
            return self._json(model_status_payload())
        if path == "/api/profiles":
            return self._json(list_profiles())
        if path.startswith("/api/session/"):
            sid = path.rsplit("/", 1)[-1]
            if sid not in SESSIONS:
                return self._json({"error": "会话不存在或服务已重启"}, 404)
            return self._json(session_payload(sid))
        rel = "index.html" if path == "/" else path.lstrip("/")
        target = (WEB / rel).resolve()
        if not str(target).startswith(str(WEB)) or not target.is_file():
            return self._json({"error": "找不到该资源"}, 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        return self._send(200, target.read_bytes(), ctype)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self._json({"error": "请求长度无效"}, 400)
        if length < 0 or length > MAX_REQUEST_BYTES:
            return self._json({"error": "请求超过 28 MiB 限制"}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "请求体不是合法 JSON"}, 400)

        if self.path == "/api/run":
            pid = body.get("profile_id", "P-A")
            try:
                profile = load_profile(pid)
            except FileNotFoundError:
                return self._json({"error": f"没有画像 {pid}"}, 404)
            orch = Orchestrator(llm=get_model_client())
            session = orch.run(profile, max_kp=int(body.get("max_kp", 3)))
            sid = uuid.uuid4().hex[:12]
            SESSIONS[sid] = (orch, session)
            return self._json(session_payload(sid))

        if self.path == "/api/intake":
            text = body.get("text", "")
            if not isinstance(text, str) or len(text.strip()) < 8:
                return self._json({"error": "请至少写 8 个字符的学习情况"}, 400)
            if len(text) > 2000:
                return self._json({"error": "学习情况不得超过 2000 个字符"}, 400)
            agent = IntakeAgent(get_model_client())
            bg = agent.parse(text)
            ex = _examiner()
            analysis = ex.analyze(bg, text) if ex else {}
            return self._json({"background": bg, "clarify": agent.clarify(bg),
                               "analysis": analysis})

        if self.path == "/api/materials/stage":
            try:
                report = stage_material(
                    body.get("filename"), body.get("content_base64"),
                    body.get("source_description"), body.get("rights_confirmed"),
                )
            except UploadError as exc:
                return self._json({"error": str(exc)}, 400)
            except Exception as exc:                       # noqa: BLE001
                # 原始文件仍会保留在 incoming，避免处理失败时静默丢失资料。
                return self._json({"error": f"暂存处理失败：{type(exc).__name__} {exc}"}, 500)
            return self._json(report, 201)

        if self.path == "/api/interview/start":
            sid = uuid.uuid4().hex[:12]
            iv = AdaptiveSession(_ITEMS, _KPS, body.get("background", {}),
                                 max_items=int(body.get("max_items", 16)),
                                 examiner=_examiner())
            INTERVIEWS[sid] = iv
            item = iv.next_item()
            return self._json({"interview_id": sid, "prior": round(iv.prior, 3),
                               "item": public_interview_item(item),
                               "snapshot": iv.snapshot()})

        if self.path == "/api/interview/answer":
            iid = body.get("interview_id")
            iv = INTERVIEWS.get(iid)
            if iv is None:
                return self._json({"error": "访谈会话不存在或服务已重启"}, 404)
            try:
                step = iv.answer(body["item_id"], int(body["choice"]))
            except (KeyError, ValueError) as exc:
                return self._json({"error": f"作答无效：{exc}"}, 400)
            nxt = iv.next_item()
            return self._json({"step": step,
                               "item": public_interview_item(nxt),
                               "snapshot": iv.snapshot()})

        if self.path == "/api/interview/finish":
            iid = body.get("interview_id")
            iv = INTERVIEWS.get(iid)
            if iv is None:
                return self._json({"error": "访谈会话不存在或服务已重启"}, 404)
            profile = {"id": f"LIVE-{iid[:6]}", "name": "本次访谈",
                       "background": body.get("background", {}),
                       "responses": iv.responses()}
            orch = Orchestrator(llm=get_model_client())
            session = orch.run(profile, max_kp=int(body.get("max_kp", 4)))
            sid = uuid.uuid4().hex[:12]
            SESSIONS[sid] = (orch, session)
            payload = session_payload(sid)
            payload["interview"] = iv.snapshot()
            ex = _examiner()
            if ex is not None:
                payload["synthesis"] = ex.synthesize(session.diagnosis, iv.log)
            return self._json(payload)

        if self.path == "/api/feedback":
            sid = body.get("session_id")
            if sid not in SESSIONS:
                return self._json({"error": "会话不存在或服务已重启"}, 404)
            orch, session = SESSIONS[sid]
            try:
                answers, feedback_result = score_feedback_choices(
                    session, body.get("kp"), body.get("choices"))
                decision = orch.feedback(session, body["kp"], answers)
            except Exception as exc:                      # noqa: BLE001
                return self._json({"error": str(exc)}, 400)
            payload = session_payload(sid)
            payload["decision"] = decision
            payload["feedback_result"] = feedback_result
            return self._json(payload)

        return self._json({"error": "无此接口"}, 404)


def main(port: int = 8000) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"演示服务已启动：http://127.0.0.1:{port}")
    print("按 Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
