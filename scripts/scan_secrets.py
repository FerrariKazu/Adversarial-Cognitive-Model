#!/usr/bin/env python3
"""Read-only secret scan for the working tree, git history, and public HF datasets.

Never prints matched text. Findings include only file, commit/repo, pattern name,
and match length.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
MAX_TEXT_BYTES = 5 * 1024 * 1024
SKIP_SUFFIXES = {
    ".pth", ".pt", ".safetensors", ".zip", ".npz", ".npy", ".png", ".jpg",
    ".jpeg", ".gif", ".webp", ".pdf", ".bin", ".onnx", ".ckpt", ".tar",
    ".gz", ".7z", ".rar", ".mp4", ".mov", ".wav", ".mp3",
}
TEXT_SUFFIXES = {
    "", ".py", ".md", ".txt", ".json", ".jsonl", ".log", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".sh", ".ipynb", ".csv", ".tsv", ".html",
    ".js", ".ts", ".css", ".xml", ".env", ".example", ".sql", ".rst",
}
PATTERNS: Dict[str, re.Pattern[str]] = {
    "HF_TOKEN_LITERAL": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GITHUB_TOKEN_LITERAL": re.compile(r"\b(?:ghp|gho)_[A-Za-z0-9]{20,}\b"),
    "GITHUB_PAT_LITERAL": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "AWS_ACCESS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "PRIVATE_KEY_MARKER": re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
}
ASSIGNMENT_PATTERN = re.compile(
    r"\b(?P<lhs>HF_TOKEN|GITHUB_TOKEN|KAGGLE_KEY|KAGGLE_USERNAME|PASSWORD|SECRET|API_KEY|ACCESS_KEY)"
    r"\s*=\s*(?P<rhs>[^\s#]+)",
    re.IGNORECASE,
)
ENV_LOOKUP_RE = re.compile(r"^(?:os\.(?:environ|getenv)|getenv|env|settings\.|\$\{|\$[A-Za-z_])")
PLACEHOLDERS = {
    "", '""', "''", "null", "none", "~", "changeme", "change-me", "redacted",
    "placeholder", "example", "dummy", "test", "test_token", "testtoken",
    "***", "your_token_here",
}


def _run(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args), cwd=str(ROOT), capture_output=True, text=True,
        errors="replace", check=False,
    )


def _git_bytes(args: Sequence[str]) -> bytes:
    return subprocess.run(
        list(args), cwd=str(ROOT), capture_output=True, check=False
    ).stdout


def _is_scan_candidate(path: Path, rel: str) -> bool:
    if any(part in {".git", "__pycache__"} for part in Path(rel).parts):
        return False
    name = path.name
    if name == ".env" or name.startswith(".env."):
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    if path.suffix.lower() not in TEXT_SUFFIXES and "." in name:
        return False
    try:
        if path.stat().st_size >= MAX_TEXT_BYTES:
            return False
    except OSError:
        return False
    return True


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _classify_assignment(rhs: str) -> str:
    raw = rhs.strip().rstrip(",;")
    if not raw:
        return "EMPTY_OR_PLACEHOLDER"
    if ENV_LOOKUP_RE.match(raw) or "os.environ" in raw or "getenv(" in raw:
        return "ENV_LOOKUP"
    if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
        inner = raw[1:-1]
        lowered = inner.lower()
        if not inner or lowered in PLACEHOLDERS:
            return "EMPTY_OR_PLACEHOLDER"
        if any(marker in lowered for marker in ("fake", "dummy", "placeholder", "example", "redacted")):
            return "EMPTY_OR_PLACEHOLDER"
        return "QUOTED_LITERAL"
    if raw.lower() in PLACEHOLDERS:
        return "EMPTY_OR_PLACEHOLDER"
    return "OTHER"


def _is_plausible_literal(shape: str, rhs: str) -> bool:
    if shape == "QUOTED_LITERAL":
        return True
    if shape != "OTHER":
        return False
    raw = rhs.strip().rstrip(",;")
    if raw.isidentifier():
        return False
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{8,}", raw):
        return False
    return any(ch.isdigit() for ch in raw) and any(ch in "_.-" for ch in raw)


def _findings(text: str) -> Iterator[Tuple[str, int, str | None]]:
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            yield name, len(match.group(0)), None
    for match in ASSIGNMENT_PATTERN.finditer(text):
        rhs = match.group("rhs")
        shape = _classify_assignment(rhs)
        if _is_plausible_literal(shape, rhs):
            yield "TOKEN_ASSIGNMENT", len(match.group(0)), shape


def _line_hash(content: str) -> str:
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode("utf-8", "replace")).hexdigest()


def _working_tree_paths() -> List[str]:
    tracked = _git_bytes(["git", "ls-files", "-z"]).decode("utf-8", "replace")
    untracked = _git_bytes(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"]
    ).decode("utf-8", "replace")
    paths = [p for p in tracked.split("\x00") if p]
    paths += [p for p in untracked.split("\x00") if p]
    return sorted(set(paths))


def scan_working_tree() -> List[str]:
    findings: List[str] = []
    for rel in _working_tree_paths():
        path = ROOT / rel
        if not path.is_file() or not _is_scan_candidate(path, rel):
            continue
        text = _read_text(path)
        if text is None:
            continue
        for pattern_name, length, shape in _findings(text):
            extra = f" shape={shape}" if shape else ""
            findings.append(
                f"FINDING file={rel} commit=WORKTREE pattern={pattern_name} "
                f"match_len={length}{extra}"
            )
    return findings


def scan_git_history() -> List[str]:
    findings: List[str] = []
    revs = [r for r in _git_bytes(["git", "rev-list", "--all"]).decode().splitlines() if r]
    needle = (
        "hf_|ghp_|gho_|github_pat_|AKIA|PRIVATE KEY|HF_TOKEN|GITHUB_TOKEN|"
        "KAGGLE_KEY|KAGGLE_USERNAME|PASSWORD|SECRET|API_KEY|ACCESS_KEY"
    )
    seen: set[Tuple[str, str, str]] = set()
    for i in range(0, len(revs), 100):
        chunk = revs[i:i + 100]
        result = _run(["git", "grep", "-I", "-n", "-E", "-e", needle, *chunk, "--"])
        if result.returncode not in (0, 1):
            continue
        for line in result.stdout.splitlines():
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue
            commit, rel, _lineno, content = parts
            for pattern_name, length, shape in _findings(content):
                key = (rel, _line_hash(content), pattern_name)
                if key in seen:
                    continue
                seen.add(key)
                extra = f" shape={shape}" if shape else ""
                findings.append(
                    f"FINDING file={rel} commit={commit} pattern={pattern_name} "
                    f"match_len={length}{extra}"
                )
    return findings


def _doc_repo_ids() -> List[str]:
    ids = set()
    for path in ROOT.rglob("*.md"):
        if any(part in {".git", "node_modules"} for part in path.parts):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for match in re.finditer(r"FerrariKazu/[A-Za-z0-9_.-]+", text):
            repo = match.group(0).rstrip(".")
            if repo.endswith(".git"):
                repo = repo[:-4]
            if repo != "FerrariKazu/Adversarial-Cognitive-Model":
                ids.add(repo)
    ids.update({
        "FerrariKazu/rhan-nxa-g2-checkpoints",
        "FerrariKazu/rhan-nxa-g2-checkpoints-rolling",
    })
    return sorted(ids)


def scan_hf_datasets() -> Tuple[List[str], List[str]]:
    findings: List[str] = []
    notes: List[str] = []
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:
        return findings, [f"HF_SKIP reason=missing_huggingface_hub detail={type(exc).__name__}"]
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    for repo_id in _doc_repo_ids():
        try:
            info = api.dataset_info(repo_id=repo_id, files_metadata=True)
        except Exception as exc:
            notes.append(f"HF_SKIP repo={repo_id} reason={type(exc).__name__}")
            continue
        for sibling in info.siblings:
            rel = sibling.rfilename
            size = getattr(sibling, "size", None)
            suffix = Path(rel).suffix.lower()
            if size is None or size >= MAX_TEXT_BYTES:
                continue
            if suffix not in TEXT_SUFFIXES and suffix:
                continue
            try:
                local = hf_hub_download(
                    repo_id=repo_id, filename=rel, repo_type="dataset", token=token
                )
                text = _read_text(Path(local))
            except Exception as exc:
                notes.append(f"HF_SKIP repo={repo_id} file={rel} reason={type(exc).__name__}")
                continue
            if text is None:
                continue
            for pattern_name, length, shape in _findings(text):
                extra = f" shape={shape}" if shape else ""
                findings.append(
                    f"FINDING file={repo_id}:{rel} commit=HF pattern={pattern_name} "
                    f"match_len={length}{extra}"
                )
    return findings, notes


def env_status() -> List[str]:
    ignored = _run(["git", "check-ignore", "-q", ".env"]).returncode == 0
    tracked = _run(["git", "ls-files", "--error-unmatch", ".env"]).returncode == 0
    history = _run(["git", "log", "--all", "--oneline", "--", ".env"]).stdout.strip()
    return [
        f"ENV_STATUS ignored={str(ignored).lower()} tracked={str(tracked).lower()} "
        f"history_commits={len(history.splitlines()) if history else 0}"
    ]


def main() -> int:
    findings: List[str] = []
    findings.extend(scan_working_tree())
    findings.extend(scan_git_history())
    hf_findings, hf_notes = scan_hf_datasets()
    findings.extend(hf_findings)
    status_lines = env_status() + hf_notes
    if findings:
        print("SECRETS FOUND")
        for line in findings:
            print(line)
        for line in status_lines:
            print(line)
        return 1
    print("NO SECRETS FOUND")
    for line in status_lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
