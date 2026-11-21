"""Hugging Face rolling/best artifact synchronization.

Stage 0A uses dedicated repos, never the RHAN-NXA, RHAN-IM, or RHAN-Bi
namespaces.  Every checkpoint has a SHA-256 sidecar; a resume is refused if the
sidecar is missing, inaccessible, or inconsistent.

Uploads are synchronous and verified before the trainer advances.  This is
slightly slower than fire-and-forget uploads but satisfies the requirement that
final uploads have completed before a run is declared complete.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stage0a.config import HF_BEST_REPO, HF_FEATURE_REPOS, HF_ROLLING_REPO, hf_feature_repo
from stage0a.utils import atomic_write_text, resolve_hf_token, sha256_file


class HFUnavailable(RuntimeError):
    """The remote state could not be verified; never fall through to fresh."""


class HFProvenanceError(RuntimeError):
    """Remote and local artifacts disagree."""


@dataclass
class DownloadedArtifact:
    local_path: Path
    remote_name: str
    sha256: str
    source: str


class HFStore:
    def __init__(
        self,
        token: Optional[str] = None,
        best_repo: str = HF_BEST_REPO,
        rolling_repo: str = HF_ROLLING_REPO,
        *,
        enabled: bool = True,
    ):
        self.token = token or resolve_hf_token()
        self.best_repo = best_repo
        self.rolling_repo = rolling_repo
        self.enabled = enabled
        self.api = None
        if self.enabled:
            if not self.token:
                raise HFUnavailable("HF_TOKEN is required for Stage 0A durability")
            from huggingface_hub import HfApi

            self.api = HfApi(token=self.token)

    # ── repositories ───────────────────────────────────────────────────────
    def feature_repo(self, backbone: str) -> str:
        return hf_feature_repo(backbone)

    def ensure_repo(self, repo_id: str) -> None:
        if not self.enabled:
            return
        try:
            self.api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        except Exception as e:
            raise HFUnavailable(f"cannot create/verify HF repo {repo_id}: {e}") from e

    def ensure_repos(self) -> None:
        if not self.enabled:
            return
        repos = [self.best_repo, self.rolling_repo]
        for repo in HF_FEATURE_REPOS.values():
            if repo not in repos:
                repos.append(repo)
        for repo in repos:
            self.ensure_repo(repo)

    def remote_name(self, run_id: str, filename: str) -> str:
        return f"{run_id.strip('/')}/{filename}"

    # ── existence ──────────────────────────────────────────────────────────
    def status(self, repo_id: str, remote_name: str) -> str:
        """Return 'present', 'absent', or raise HFUnavailable.

        A false is only trusted after a successful API call.  Network/auth
        failures are never interpreted as "no checkpoint".
        """
        if not self.enabled:
            return "absent"
        try:
            exists = self.api.file_exists(repo_id=repo_id, filename=remote_name, repo_type="dataset")
            return "present" if exists else "absent"
        except Exception as e:
            raise HFUnavailable(f"cannot verify {repo_id}/{remote_name}: {e}") from e

    # ── download ───────────────────────────────────────────────────────────
    def download_verified(
        self,
        repo_id: str,
        remote_name: str,
        local_path: str | Path,
        *,
        expected_sha: Optional[str] = None,
    ) -> DownloadedArtifact:
        if not self.enabled:
            raise HFUnavailable("HF synchronization is disabled")
        from huggingface_hub import hf_hub_download

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_name = remote_name + ".sha256"
        try:
            sidecar_path = hf_hub_download(
                repo_id=repo_id, filename=sidecar_name, repo_type="dataset", token=self.token
            )
            remote_sha = Path(sidecar_path).read_text(encoding="utf-8").strip().split()[0]
        except Exception as e:
            raise HFProvenanceError(
                f"missing/unreadable SHA-256 sidecar for {repo_id}/{remote_name}: {e}"
            ) from e
        if expected_sha and remote_sha != expected_sha:
            raise HFProvenanceError(
                f"SHA-256 sidecar mismatch for {repo_id}/{remote_name}: "
                f"remote={remote_sha} expected={expected_sha}"
            )
        try:
            cached = hf_hub_download(
                repo_id=repo_id, filename=remote_name, repo_type="dataset", token=self.token
            )
        except Exception as e:
            raise HFUnavailable(f"cannot download {repo_id}/{remote_name}: {e}") from e
        local_sha = sha256_file(cached)
        if local_sha != remote_sha:
            raise HFProvenanceError(
                f"downloaded artifact hash mismatch for {repo_id}/{remote_name}: "
                f"file={local_sha} sidecar={remote_sha}"
            )
        tmp = local_path.with_suffix(local_path.suffix + ".download")
        shutil.copyfile(cached, tmp)
        os.replace(tmp, local_path)
        atomic_write_text(str(local_path) + ".sha256", remote_sha + "\n")
        return DownloadedArtifact(local_path=local_path, remote_name=remote_name, sha256=remote_sha, source=repo_id)

    # ── upload ─────────────────────────────────────────────────────────────
    def upload_verified(
        self,
        local_path: str | Path,
        repo_id: str,
        remote_name: str,
        *,
        required: bool = True,
        extra_files: Optional[List[Tuple[str | Path, str]]] = None,
    ) -> Optional[DownloadedArtifact]:
        """Upload artifact + sidecar (plus optional extra files) in a single atomic commit."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        local_sha = sha256_file(local_path)
        sidecar = local_path.with_suffix(local_path.suffix + ".sha256")
        atomic_write_text(sidecar, local_sha + "\n")
        if not self.enabled:
            if required:
                raise HFUnavailable("HF upload required but synchronization is disabled")
            return None

        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(path_in_repo=remote_name, path_or_fileobj=str(local_path)),
            CommitOperationAdd(path_in_repo=remote_name + ".sha256", path_or_fileobj=str(sidecar)),
        ]
        if extra_files:
            for extra_p, extra_rem in extra_files:
                ep = Path(extra_p)
                if ep.exists():
                    operations.append(CommitOperationAdd(path_in_repo=extra_rem, path_or_fileobj=str(ep)))

        max_retries = 3
        committed = False
        for attempt in range(max_retries):
            try:
                self.api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Sync {remote_name}",
                    token=self.token,
                )
                committed = True
                break
            except Exception as e:
                err_str = str(e)
                is_429 = "429" in err_str or "rate limit" in err_str.lower() or getattr(getattr(e, "response", None), "status_code", None) == 429
                if is_429 and attempt < max_retries - 1:
                    wait_s = min(30, 5 * (2 ** attempt))
                    print(f"[HFStore] 429 rate limit, waiting {wait_s}s before retry (attempt {attempt+1}/{max_retries})...", flush=True)
                    time.sleep(wait_s)
                    continue
                if not required:
                    print(f"[HFStore] Warning: upload failed for {repo_id}/{remote_name}: {e}; continuing with local durable copy", flush=True)
                    return None
                raise HFUnavailable(f"upload failed for {repo_id}/{remote_name}: {e}") from e

        if not committed:
            if not required:
                return None
            raise HFUnavailable(f"upload failed for {repo_id}/{remote_name}")

        # Verify size and, when the hub exposes a sha256 ETag (LFS), the digest.
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            meta = get_hf_file_metadata(
                hf_hub_url(repo_id=repo_id, filename=remote_name, repo_type="dataset"),
                token=self.token,
            )
            if int(meta.size) != local_path.stat().st_size:
                raise HFProvenanceError(
                    f"remote size mismatch for {repo_id}/{remote_name}: "
                    f"remote={meta.size} local={local_path.stat().st_size}"
                )
            etag = (meta.etag or "").strip()
            if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag.lower()):
                if etag.lower() != local_sha:
                    raise HFProvenanceError(
                        f"remote ETag mismatch for {repo_id}/{remote_name}: "
                        f"remote={etag} local={local_sha}"
                    )
        except HFProvenanceError:
            raise
        except Exception as e:
            if not required:
                print(f"[HFStore] Warning: remote verification failed for {repo_id}/{remote_name}: {e}", flush=True)
                return DownloadedArtifact(local_path=local_path, remote_name=remote_name, sha256=local_sha, source=repo_id)
            raise HFUnavailable(f"cannot verify upload {repo_id}/{remote_name}: {e}") from e
        return DownloadedArtifact(local_path=local_path, remote_name=remote_name, sha256=local_sha, source=repo_id)

    def upload_artifact(self, local_path: str | Path, repo_id: str, remote_name: str, *, required: bool = True):
        return self.upload_verified(local_path, repo_id, remote_name, required=required)

    # ── rolling restore ────────────────────────────────────────────────────
    def restore_rolling(
        self,
        run_id: str,
        local_path: str | Path,
        *,
        force_fresh: bool = False,
    ) -> Tuple[Optional[Path], str]:
        """Return (path, source). Raises rather than silently restarting.

        source is one of: 'remote', 'local', 'fresh'.
        """
        local_path = Path(local_path)
        remote_name = self.remote_name(run_id, "rolling.pt")
        if force_fresh:
            return None, "fresh"
        remote_state = self.status(self.rolling_repo, remote_name)
        local_state = "present" if local_path.exists() else "absent"

        if remote_state == "present":
            # Always refresh: a Kaggle restart may leave an older local copy.
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.download_verified(self.rolling_repo, remote_name, local_path)
            return local_path, "remote"
        if local_state == "present":
            sidecar = Path(str(local_path) + ".sha256")
            if not sidecar.exists():
                raise HFProvenanceError(
                    f"local rolling checkpoint {local_path} has no SHA-256 sidecar; "
                    "refusing an unverifiable resume"
                )
            expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
            actual = sha256_file(local_path)
            if expected != actual:
                raise HFProvenanceError(
                    f"local rolling checkpoint hash mismatch: {actual} != {expected}"
                )
            return local_path, "local"
        return None, "fresh"

    def sync_rolling(
        self,
        run_id: str,
        local_path: str | Path,
        log_path: Optional[str | Path] = None,
        *,
        required: bool = False,
    ) -> None:
        extra = []
        if log_path is not None and Path(log_path).exists():
            extra.append((log_path, self.remote_name(run_id, "metrics.jsonl")))
        self.upload_verified(
            local_path,
            self.rolling_repo,
            self.remote_name(run_id, "rolling.pt"),
            required=required,
            extra_files=extra,
        )

    def sync_best(self, run_id: str, local_path: str | Path, *, required: bool = False) -> None:
        self.upload_verified(local_path, self.best_repo, self.remote_name(run_id, "best.pt"), required=required)

    def sync_final(self, run_id: str, local_path: str | Path) -> None:
        self.upload_verified(local_path, self.best_repo, self.remote_name(run_id, "final.pt"), required=True)

    def sync_json(self, run_id: str, local_path: str | Path, filename: str, *, rolling: bool = False) -> None:
        repo = self.rolling_repo if rolling else self.best_repo
        self.upload_verified(local_path, repo, self.remote_name(run_id, filename), required=False)

    def sync_log(self, run_id: str, local_path: str | Path) -> None:
        self.upload_verified(
            local_path,
            self.rolling_repo,
            self.remote_name(run_id, "metrics.jsonl"),
            required=False,
        )


class NoopHFStore:
    """Local-only store for deterministic tests; never used by the notebooks."""

    best_repo = "local/test-best"
    rolling_repo = "local/test-rolling"
    enabled = False

    def feature_repo(self, backbone: str) -> str:
        return f"local/test-features-{backbone}"

    def ensure_repo(self, repo_id: str) -> None:
        pass

    def ensure_repos(self) -> None:
        pass

    def remote_name(self, run_id: str, filename: str) -> str:
        return f"{run_id.strip('/')}/{filename}"

    def status(self, repo_id: str, remote_name: str) -> str:
        return "absent"

    def restore_rolling(self, run_id: str, local_path: str | Path, *, force_fresh: bool = False):
        return (None, "fresh") if not Path(local_path).exists() else (Path(local_path), "local")

    def sync_rolling(self, *args, **kwargs) -> None:
        return None

    def sync_best(self, *args, **kwargs) -> None:
        return None

    def sync_final(self, *args, **kwargs) -> None:
        return None

    def sync_json(self, *args, **kwargs) -> None:
        return None

    def sync_log(self, *args, **kwargs) -> None:
        return None
