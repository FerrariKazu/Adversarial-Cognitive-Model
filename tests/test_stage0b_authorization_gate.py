import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "audit" / "stage0b_execution_v1" / "scripts"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import stage0b_runner as R  # noqa: E402


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def valid_execution_approval(path: Path, payload: str = R.APPROVED_PAYLOAD) -> None:
    write_json(
        path,
        {
            "schema": "rhan-stage0b-execution-approval/v1",
            "approved": True,
            "execution_approved": True,
            "protocol_manifest_payload_sha256": payload,
        },
    )


def valid_data_use_confirmation(
    path: Path,
    payload: str = R.APPROVED_PAYLOAD,
    canonical: str = R.APPROVED_PARTITION_CANONICAL,
    partition_json: str = R.APPROVED_PARTITION_JSON,
) -> None:
    write_json(
        path,
        {
            "schema": "rhan-stage0b-data-use-confirmation/v1",
            "confirmation_status": "confirmed",
            "data_use_confirmed": True,
            "confirmer_name": "Test Confirmer",
            "confirmer_role": "Test Supervisor",
            "confirmation_date_utc": "2026-11-22T00:00:00Z",
            "approved_scope": ["local calibration temperature fitting", "local one-shot evaluation"],
            "dataset_identity": {
                "dataset_name": "ImageNetV2 matched-frequency validation",
                "partition_canonical_sha256": canonical,
                "partition_json_sha256": partition_json,
            },
            "protocol_manifest_payload_sha256": payload,
            "restrictions": ["local-only artifacts", "no upload"],
            "publication_rights_granted": False,
            "evidence_reference": "test-evidence",
        },
    )


def test_gate_table_missing_missing_blocks(tmp_path):
    approval = tmp_path / "approval.json"
    data_use = tmp_path / "data_use.json"
    with pytest.raises(RuntimeError):
        R.validate_execution_gates(approval, data_use)


def test_gate_table_valid_missing_blocks(tmp_path):
    approval = tmp_path / "approval.json"
    data_use = tmp_path / "data_use.json"
    valid_execution_approval(approval)
    with pytest.raises(RuntimeError):
        R.validate_execution_gates(approval, data_use)


def test_gate_table_missing_valid_blocks(tmp_path):
    approval = tmp_path / "approval.json"
    data_use = tmp_path / "data_use.json"
    valid_data_use_confirmation(data_use)
    with pytest.raises(RuntimeError):
        R.validate_execution_gates(approval, data_use)


def test_gate_table_valid_valid_allows(tmp_path):
    approval = tmp_path / "approval.json"
    data_use = tmp_path / "data_use.json"
    valid_execution_approval(approval)
    valid_data_use_confirmation(data_use)
    result = R.validate_execution_gates(approval, data_use)
    assert "execution_approval" in result
    assert "data_use_confirmation" in result


def test_invalid_execution_approval_blocks(tmp_path):
    approval = tmp_path / "approval.json"
    data_use = tmp_path / "data_use.json"
    write_json(
        approval,
        {
            "schema": "rhan-stage0b-execution-approval/v1",
            "approved": False,
            "execution_approved": False,
            "protocol_manifest_payload_sha256": R.APPROVED_PAYLOAD,
        },
    )
    valid_data_use_confirmation(data_use)
    with pytest.raises(RuntimeError, match="approved and execution_approved"):
        R.validate_execution_gates(approval, data_use)


def test_invalid_data_use_confirmation_blocks(tmp_path):
    approval = tmp_path / "approval.json"
    data_use = tmp_path / "data_use.json"
    valid_execution_approval(approval)
    write_json(
        data_use,
        {
            "schema": "rhan-stage0b-data-use-confirmation/v1",
            "confirmation_status": "confirmed",
            "data_use_confirmed": True,
            "confirmer_role": "Test Supervisor",
            "confirmation_date_utc": "2026-11-22T00:00:00Z",
            "approved_scope": ["local evaluation"],
            "dataset_identity": {
                "dataset_name": "ImageNetV2 matched-frequency validation",
                "partition_canonical_sha256": R.APPROVED_PARTITION_CANONICAL,
                "partition_json_sha256": R.APPROVED_PARTITION_JSON,
            },
            "protocol_manifest_payload_sha256": R.APPROVED_PAYLOAD,
            "restrictions": ["local-only"],
            "publication_rights_granted": True,
            "evidence_reference": "test-evidence",
        },
    )
    with pytest.raises(RuntimeError, match="publication rights"):
        R.validate_execution_gates(approval, data_use)


def test_invalid_protocol_hash_blocks(monkeypatch):
    monkeypatch.setattr(R, "APPROVED_PAYLOAD", "0" * 64)
    with pytest.raises(RuntimeError):
        R.load_protocol()


def test_invalid_checkpoint_hash_blocks(monkeypatch):
    bad_heads = dict(R.APPROVED_HEADS)
    bad_heads[17] = "0" * 64
    monkeypatch.setattr(R, "APPROVED_HEADS", bad_heads)
    with pytest.raises(RuntimeError):
        R.load_protocol()


def test_invalid_partition_hash_blocks(monkeypatch):
    monkeypatch.setattr(R, "APPROVED_PARTITION_JSON", "0" * 64)
    with pytest.raises(RuntimeError):
        R.load_protocol()


def test_publication_status_does_not_block_gates(tmp_path):
    approval = tmp_path / "approval.json"
    data_use = tmp_path / "data_use.json"
    valid_execution_approval(approval)
    valid_data_use_confirmation(data_use)
    # Publication status is not an input to the execution gates.
    result = R.validate_execution_gates(approval, data_use)
    assert result is not None


def test_score_requires_fit_complete(tmp_path, monkeypatch):
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    monkeypatch.setattr(R, "EXEC_DIR", exec_dir)
    monkeypatch.setattr(R, "DATA_USE_CONFIRMATION_PATH", exec_dir / "data_use_confirmation.json")
    approval = exec_dir / "execution_approval.json"
    data_use = exec_dir / "data_use_confirmation.json"
    valid_execution_approval(approval)
    valid_data_use_confirmation(data_use)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="fit_complete.json missing"):
        R.cmd_score(type("Args", (), {"out": str(run_dir), "force": False})())


def test_score_requires_temperatures(tmp_path, monkeypatch):
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    monkeypatch.setattr(R, "EXEC_DIR", exec_dir)
    monkeypatch.setattr(R, "DATA_USE_CONFIRMATION_PATH", exec_dir / "data_use_confirmation.json")
    approval = exec_dir / "execution_approval.json"
    data_use = exec_dir / "data_use_confirmation.json"
    valid_execution_approval(approval)
    valid_data_use_confirmation(data_use)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "fit_complete.json").write_text("{}\n")
    with pytest.raises((RuntimeError, FileNotFoundError, json.JSONDecodeError)):
        R.cmd_score(type("Args", (), {"out": str(run_dir), "force": False})())
