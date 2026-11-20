"""Small, dependency-light utilities shared by the Stage 0A pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    """Append one JSON line and fsync; safe against a session timeout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def git_commit(short: bool = False, cwd: Optional[str] = None) -> str:
    args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except Exception:
        pass
    return "unknown"


def file_set_fingerprint(paths: Iterable[str | Path]) -> str:
    """Hash file paths and contents for the training-code fingerprint."""
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(str(p).encode())
        h.update(b"\0")
        if p.exists():
            h.update(sha256_file(p).encode())
        else:
            h.update(b"MISSING")
        h.update(b"\n")
    return h.hexdigest()


def training_fingerprint(repo_root: str | Path) -> str:
    """Identity of the code that can change training semantics.

    Notebook-only or documentation-only commits must not invalidate a resume.
    We therefore hash the Stage 0A package and the two launchers, plus the Git
    commit as a secondary provenance field.
    """
    root = Path(repo_root)
    paths = sorted((root / "stage0a").glob("*.py"))
    paths += [root / "cloud_setup" / "Kaggle_STAGE0A.py", root / "cloud_setup" / "colab_STAGE0A.py"]
    return file_set_fingerprint([p for p in paths if p.exists()])


def state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    """Deterministic hash of a model state dict (dtype/shape/bytes included)."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            continue
        t = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    # Kaggle secret (optional import).
    try:
        from kaggle_secrets import UserSecretsClient

        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
            return tok
    except Exception:
        pass
    # Repository-local .env, never printed.
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env = parent / ".env"
        if env.exists():
            try:
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("HF_TOKEN"):
                        _, _, value = line.partition("=")
                        value = value.strip().strip('"').strip("'")
                        if value:
                            return value
            except Exception:
                pass
            break
    return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    cuda = state.get("torch_cuda") or []
    if cuda and torch.cuda.is_available():
        # A checkpoint from a 2-GPU run is resumed on the same visible-device
        # layout; refusing is safer than restoring a partial RNG set.
        if len(cuda) != torch.cuda.device_count():
            raise RuntimeError(
                f"RNG device mismatch: checkpoint has {len(cuda)} CUDA RNG states, "
                f"current process sees {torch.cuda.device_count()} device(s). "
                "Resume with the same visible-device layout."
            )
        torch.cuda.set_rng_state_all(cuda)


def json_load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def human_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def disk_usage(path: str | Path) -> Dict[str, Any]:
    import shutil

    usage = shutil.disk_usage(str(path))
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def read_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default
