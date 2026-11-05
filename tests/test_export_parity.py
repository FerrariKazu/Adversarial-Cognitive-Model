from pathlib import Path

import pytest

from scripts.export_safetensors import OFFICIAL_CHECKPOINT, export_checkpoint


@pytest.mark.skipif(
    not OFFICIAL_CHECKPOINT.exists(),
    reason=f"official checkpoint not present: {OFFICIAL_CHECKPOINT}",
)
def test_official_export_cpu_eval_parity_is_exact(tmp_path):
    result = export_checkpoint(OFFICIAL_CHECKPOINT, tmp_path / "export")
    assert result["parity_max_abs"] == 0.0
    assert result["param_count"] > 0
    assert Path(result["weights_path"]).is_file()
    assert Path(result["config_path"]).is_file()
