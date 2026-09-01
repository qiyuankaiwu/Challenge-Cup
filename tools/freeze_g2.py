"""冻结 G2 可复现证据包。

运行示例（从仓库根目录）：

    $env:PYTHONUTF8='1'
    py -3 -m tools.freeze_g2

该工具只记录自动化自检证据。它不会伪造知识来源核验、领域真值或人工盲评；
这些项目会在生成的 README 中继续标为待人工完成。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "06_测试数据与评测报告"
EVAL_SEED = 20260905


def _run(command: list[str], log_path: Path) -> None:
    """运行命令，完整保留 stdout/stderr；失败时停止冻结。"""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + subprocess.list2cmdline(command) + "\n\n" + result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"命令失败（退出码 {result.returncode}）：{' '.join(command)}")


def _output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else f"不可用：{result.stderr.strip()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_readme(out: Path) -> None:
    (out / "README.md").write_text(
        """# G2 自动化证据冻结包

本目录由 `py -3 -m tools.freeze_g2` 生成。`基线与复现信息.json` 记录运行时、Git SHA、输入哈希与命令；每项命令的完整控制台输出均在 `测试通过记录/`。

## 证据索引

- `50组以上测试明细/完整流程/`：固定 50 组的原始案例、断言明细及自动汇总。
- `消融与红队报告/审核关闭/`、`辩论关闭/`：和完整流程使用同一代码、数据及固定种子的对照。
- `消融与红队报告/redteam/`：H1–H6 检出率、误伤率和逐条明细。
- `题库质量报告/`：答案位置、结构瑕疵和待改题清单；没有真实作答数据时不包含难度或区分度标定。
- `测试通过记录/`：单元测试、Python/浏览器规则一致性和 JavaScript 语法检查日志。

## 严格边界

这是一份**自动化系统自检**证据，不是正式效果结论：

1. `run_eval` 的“幻觉率”是对同一知识库的自动复核，只有在所有 Demo 切片完成真实来源定位和人工核实后才可作为辅助证据。
2. `run_eval` 的“适配规则一致性”不能替代独立难度适配真值。
3. G2 仍须人工补齐：切片来源页码/章节与核实记录、两名独立评分者的盲评原表、Kappa、仲裁记录，以及三项指标的人工真值报告。
4. 红队 H5/H6 的漏检必须如实保留，不得只摘录总检出率。
""",
        encoding="utf-8",
    )


def freeze(out: Path, resume: bool = False) -> None:
    """运行全部自动化 G2 项并把环境与输入固定到一个目录。"""
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"输出目录非空，为避免覆盖证据已停止：{out}")
    out.mkdir(parents=True, exist_ok=True)

    tests = out / "测试通过记录"
    def run_once(command: list[str], log_path: Path) -> None:
        if resume and log_path.exists():
            return
        _run(command, log_path)

    run_once([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
             tests / "单元测试.log")
    run_once([sys.executable, "-m", "unittest", "tests.test_parity", "-v"],
             tests / "规则一致性.log")
    run_once(["node", "--check", "web/engine.js"], tests / "engine语法检查.log")

    full = out / "50组以上测试明细" / "完整流程"
    run_once([sys.executable, "-m", "evalkit.run_eval", "--n", "50", "--out", str(full)],
             full / "运行日志.log")
    no_audit = out / "消融与红队报告" / "审核关闭"
    run_once([sys.executable, "-m", "evalkit.run_eval", "--n", "50", "--no-audit",
              "--out", str(no_audit)], no_audit / "运行日志.log")
    no_debate = out / "消融与红队报告" / "辩论关闭"
    run_once([sys.executable, "-m", "evalkit.run_eval", "--n", "50", "--no-debate",
              "--out", str(no_debate)], no_debate / "运行日志.log")
    redteam = out / "消融与红队报告" / "redteam"
    run_once([sys.executable, "-m", "evalkit.redteam", "--out", str(redteam)],
             redteam / "运行日志.log")
    items = out / "题库质量报告"
    run_once([sys.executable, "-m", "evalkit.itemreport", "--out", str(items)],
             items / "运行日志.log")

    tracked = [
        ROOT / "config.py",
        ROOT / "data/kb/robotics.jsonl",
        ROOT / "data/pretest.json",
        ROOT / "data/profiles/P-A.json",
        ROOT / "data/profiles/P-B.json",
        ROOT / "data/profiles/P-C.json",
        ROOT / "evalkit/run_eval.py",
        ROOT / "evalkit/redteam.py",
        ROOT / "evalkit/itemreport.py",
    ]
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _output(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": _output(["git", "status", "--porcelain"]),
        "python": sys.version,
        "node": _output(["node", "--version"]),
        "environment": {"PYTHONUTF8": "1"},
        "evaluation": {"cases": 50, "seed": EVAL_SEED},
        "input_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in tracked},
        "limitations": [
            "自动幻觉复核仅证明与当前知识库的一致性，不构成独立领域真值。",
            "自动难度校验复用系统规则，不构成独立适配效果。",
            "本证据包不含人工来源核验、双盲评分、Kappa 或仲裁记录。",
        ],
    }
    (out / "基线与复现信息.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_readme(out)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="冻结 G2 自动化证据包")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true",
                        help="只补跑没有运行日志的步骤，不覆盖已有证据")
    args = parser.parse_args()
    freeze(args.out.resolve(), resume=args.resume)
    print(f"G2 自动化证据已冻结：{args.out.resolve()}")


if __name__ == "__main__":
    main()
