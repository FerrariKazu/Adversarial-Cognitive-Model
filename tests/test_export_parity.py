import json
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
    assert result["num_state_dict_values"] == 23_595_749
    assert result["num_unique_parameters"] == 23_369_573
    assert result["num_trainable_parameters"] == 23_369_573
    assert Path(result["weights_path"]).is_file()
    assert Path(result["config_path"]).is_file()
    config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert config["num_state_dict_values"] == 23_595_749
    assert config["num_unique_parameters"] == 23_369_573
    assert config["num_trainable_parameters"] == 23_369_573
    assert len(config["class_names"]) == 100
    assert config["class_names_source"]
