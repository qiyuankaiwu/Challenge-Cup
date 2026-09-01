"""Reusable primitives for reproducible, auditable evidence packages.

The module deliberately separates *recording* an automated check from claiming
that a result is a human-verified truth.  It is standard-library only so that
the package can be inspected and reproduced in an offline evaluation setting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
CHUNK_SIZE = 1024 * 1024


class EvidenceError(RuntimeError):
    """Raised when an evidence package is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class CheckSpec:
    """One command whose complete result is retained in a package."""

    id: str
    command: tuple[str, ...]
    log: str
    description: str
    required_outputs: tuple[str, ...] = ()
    timeout_seconds: int = 600

    def __post_init__(self) -> None:
        if not self.id or any(char.isspace() for char in self.id):
            raise ValueError("check id must be a non-empty token")
        if not self.command:
            raise ValueError("check command cannot be empty")
        if Path(self.log).is_absolute() or ".." in Path(self.log).parts:
            raise ValueError("log must be a package-relative path")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class CheckResult:
    """Immutable command result represented in the final manifest."""

    id: str
    command: list[str]
    log: str
    started_at: str
    finished_at: str
    duration_seconds: float
    return_code: int
    status: str
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass
class PackageManifest:
    """JSON-safe evidence-package inventory with explicit limitations."""

    created_at: str
    root: str
    source_revision: str
    environment: dict[str, str]
    inputs: dict[str, str]
    checks: list[CheckResult]
    limitations: list[str]
    schema_version: int = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checks"] = [asdict(result) for result in self.checks]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PackageManifest":
        required = {"created_at", "root", "source_revision", "environment",
                    "inputs", "checks", "limitations"}
        missing = required - set(data)
        if missing:
            raise EvidenceError(f"manifest missing fields: {', '.join(sorted(missing))}")
        if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise EvidenceError("unsupported manifest schema")
        checks = [CheckResult(**record) for record in data["checks"]]
        return cls(
            created_at=data["created_at"], root=data["root"],
            source_revision=data["source_revision"],
            environment=dict(data["environment"]), inputs=dict(data["inputs"]),
            checks=checks, limitations=list(data["limitations"]),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative_path(root: Path, path: Path) -> Path:
    """Return a safe relative path or reject paths outside the package."""
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise EvidenceError(f"path escapes package root: {path}") from exc


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks; suitable for large evaluation outputs."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_inputs(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    """Create a stable relative-path to SHA-256 mapping for source inputs."""
    result: dict[str, str] = {}
    for path in sorted({candidate.resolve() for candidate in paths}):
        if not path.is_file():
            raise EvidenceError(f"required input is missing: {path}")
        result[relative_path(root, path).as_posix()] = sha256_file(path)
    return result


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write UTF-8 JSON atomically so interrupted runs never leave partial data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent,
                                     encoding="utf-8", suffix=".tmp") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def read_manifest(path: Path) -> PackageManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise EvidenceError("manifest root must be an object")
    return PackageManifest.from_dict(raw)


def command_text(command: Sequence[str]) -> str:
    """Render a command for logs without sending it through a shell."""
    return subprocess.list2cmdline(list(command))


class EvidenceRunner:
    """Runs declarative checks and persists logs before returning each result."""

    def __init__(self, root: Path, package: Path, *, environment: dict[str, str] | None = None):
        self.root = root.resolve()
        self.package = package.resolve()
        self.environment = dict(environment or {})
        self.results: list[CheckResult] = []

    def _log_path(self, spec: CheckSpec) -> Path:
        path = self.package / spec.log
        relative_path(self.package, path)
        return path

    def _command(self, spec: CheckSpec) -> list[str]:
        return [part.replace("{python}", sys.executable) for part in spec.command]

    def run(self, spec: CheckSpec, *, resume: bool = False) -> CheckResult:
        log_path = self._log_path(spec)
        if resume and log_path.exists():
            return self._resumed(spec, log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(self.environment)
        env.setdefault("PYTHONUTF8", "1")
        command = self._command(spec)
        started = datetime.now(timezone.utc)
        error = None
        try:
            completed = subprocess.run(
                command, cwd=self.root, env=env, text=True, encoding="utf-8",
                errors="replace", capture_output=True, timeout=spec.timeout_seconds,
                check=False,
            )
            return_code = completed.returncode
            body = completed.stdout + completed.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            return_code = -1
            error = str(exc)
            body = error
        finished = datetime.now(timezone.utc)
        log_path.write_text(f"$ {command_text(command)}\n\n{body}", encoding="utf-8")
        outputs = self._hash_outputs(spec)
        status = "passed" if return_code == 0 and not error else "failed"
        result = CheckResult(
            id=spec.id, command=command, log=spec.log, started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            duration_seconds=round((finished - started).total_seconds(), 3),
            return_code=return_code, status=status, outputs=outputs, error=error,
        )
        self.results.append(result)
        return result

    def _resumed(self, spec: CheckSpec, log_path: Path) -> CheckResult:
        result = CheckResult(
            id=spec.id, command=self._command(spec), log=spec.log, started_at=utc_now(),
            finished_at=utc_now(), duration_seconds=0.0, return_code=0, status="resumed",
            outputs=self._hash_outputs(spec), error=None,
        )
        self.results.append(result)
        return result

    def _hash_outputs(self, spec: CheckSpec) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for item in spec.required_outputs:
            path = self.package / item
            if not path.is_file():
                continue
            outputs[item] = sha256_file(path)
        return outputs

    def run_all(self, specs: Iterable[CheckSpec], *, resume: bool = False,
                fail_fast: bool = True) -> list[CheckResult]:
        seen: set[str] = set()
        for spec in specs:
            if spec.id in seen:
                raise EvidenceError(f"duplicate check id: {spec.id}")
            seen.add(spec.id)
            result = self.run(spec, resume=resume)
            if not result.passed and result.status != "resumed" and fail_fast:
                raise EvidenceError(f"check failed: {spec.id}; see {result.log}")
        return list(self.results)


def validate_package(package: Path, *, verify_outputs: bool = True) -> list[str]:
    """Return all package defects instead of failing at the first one."""
    package = package.resolve()
    issues: list[str] = []
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        return ["missing manifest.json"]
    try:
        manifest = read_manifest(manifest_path)
    except EvidenceError as exc:
        return [str(exc)]
    ids: set[str] = set()
    for check in manifest.checks:
        if check.id in ids:
            issues.append(f"duplicate check id in manifest: {check.id}")
        ids.add(check.id)
        log_path = package / check.log
        try:
            relative_path(package, log_path)
        except EvidenceError as exc:
            issues.append(str(exc))
            continue
        if not log_path.is_file():
            issues.append(f"missing log for {check.id}: {check.log}")
        if check.status not in {"passed", "failed", "resumed"}:
            issues.append(f"invalid status for {check.id}: {check.status}")
        if verify_outputs:
            for name, expected in check.outputs.items():
                output = package / name
                if not output.is_file():
                    issues.append(f"missing declared output: {name}")
                elif sha256_file(output) != expected:
                    issues.append(f"output hash mismatch: {name}")
    return issues


def summarize(manifest: PackageManifest) -> dict[str, Any]:
    """Produce a small review-friendly status summary without inventing claims."""
    counts = {"passed": 0, "failed": 0, "resumed": 0}
    duration = 0.0
    for result in manifest.checks:
        counts[result.status] = counts.get(result.status, 0) + 1
        duration += result.duration_seconds
    return {
        "schema_version": manifest.schema_version,
        "created_at": manifest.created_at,
        "source_revision": manifest.source_revision,
        "checks": len(manifest.checks),
        "statuses": counts,
        "duration_seconds": round(duration, 3),
        "input_files": len(manifest.inputs),
        "limitations": list(manifest.limitations),
    }


def copy_package(source: Path, destination: Path, *, overwrite: bool = False) -> None:
    """Copy a validated evidence package only after its integrity is checked."""
    issues = validate_package(source)
    if issues:
        raise EvidenceError("cannot copy invalid package: " + "; ".join(issues))
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists: {destination}")
        if not destination.is_dir():
            raise EvidenceError("destination exists and is not a directory")
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect an evidence package")
    parser.add_argument("package", type=Path)
    parser.add_argument("--no-hash", action="store_true", help="skip output hash checks")
    args = parser.parse_args()
    issues = validate_package(args.package, verify_outputs=not args.no_hash)
    if issues:
        for issue in issues:
            print(f"INVALID: {issue}")
        raise SystemExit(1)
    print(json.dumps(summarize(read_manifest(args.package / "manifest.json")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
