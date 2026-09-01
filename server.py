"""演示服务。

    python3 server.py        然后打开 http://127.0.0.1:8000

用标准库 http.server，不装 FastAPI。骨架阶段的目标是"评委机器上 clone 下来
直接 python3 server.py 就能跑"，少一个依赖少一个现场翻车的可能。
并发和鉴权后面真要上再换 FastAPI，接口路径不变。

会话存在内存里，进程重启就没了。演示够用，别拿去当生产。
"""

from __future__ import annotations

import json
import mimetypes
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config
from agents.examiner import ExaminerAgent
from agents.intake import IntakeAgent
from core.cat import AdaptiveSession
from core.llm import build_llm
from core.retrieval import Retriever
from orchestrator import Orchestrator, load_profile

WEB = Path(__file__).resolve().parent / "web"
SESSIONS: dict[str, tuple[Orchestrator, object]] = {}
INTERVIEWS: dict[str, AdaptiveSession] = {}
_ITEMS = json.loads(config.PRETEST_PATH.read_text(encoding="utf-8"))["items"]
_KPS = json.loads(config.KP_PATH.read_text(encoding="utf-8"))["points"]
_KP_INDEX = {k["id"]: k for k in _KPS}


def _examiner():
    if not config.EXAMINER_ENABLED:
        return None
    return ExaminerAgent(build_llm(), Retriever.from_jsonl(config.KB_PATH), _KP_INDEX)


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
    return data


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
        length = int(self.headers.get("Content-Length", 0))
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
            orch = Orchestrator()
            session = orch.run(profile, max_kp=int(body.get("max_kp", 3)))
            sid = uuid.uuid4().hex[:12]
            SESSIONS[sid] = (orch, session)
            return self._json(session_payload(sid))

        if self.path == "/api/intake":
            text = body.get("text", "")
            agent = IntakeAgent(build_llm())
            bg = agent.parse(text)
            ex = _examiner()
            analysis = ex.analyze(bg, text) if ex else {}
            return self._json({"background": bg, "clarify": agent.clarify(bg),
                               "analysis": analysis})

        if self.path == "/api/interview/start":
            sid = uuid.uuid4().hex[:12]
            iv = AdaptiveSession(_ITEMS, _KPS, body.get("background", {}),
                                 max_items=int(body.get("max_items", 16)),
                                 examiner=_examiner())
            INTERVIEWS[sid] = iv
            item = iv.next_item()
            return self._json({"interview_id": sid, "prior": round(iv.prior, 3),
                               "item": item, "snapshot": iv.snapshot()})

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
            return self._json({"step": step, "item": nxt,
                               "snapshot": iv.snapshot()})

        if self.path == "/api/interview/finish":
            iid = body.get("interview_id")
            iv = INTERVIEWS.get(iid)
            if iv is None:
                return self._json({"error": "访谈会话不存在或服务已重启"}, 404)
            profile = {"id": f"LIVE-{iid[:6]}", "name": "本次访谈",
                       "background": body.get("background", {}),
                       "responses": iv.responses()}
            orch = Orchestrator()
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
            answers = [bool(a) for a in body.get("answers", [])]
            try:
                decision = orch.feedback(session, body["kp"], answers)
            except Exception as exc:                      # noqa: BLE001
                return self._json({"error": str(exc)}, 400)
            payload = session_payload(sid)
            payload["decision"] = decision
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
    finally:
        srv.server_close()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
