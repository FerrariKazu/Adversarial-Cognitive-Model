from pydantic import BaseModel
from typing import Optional


class InferenceRequest(BaseModel):
    model_id: str
    attack_eps: Optional[float] = None
    attack_steps: Optional[int] = None
    use_cpu: bool = False


class InferenceResult(BaseModel):
    model_id: str
    predicted_class: str
    predicted_idx: int
    confidence: float
    all_probs: list[float]
    attack_eps: Optional[float] = None
    correct: Optional[bool] = None


class BatchEvalRequest(BaseModel):
    model_ids: list[str]
    eps_grid: list[float]
    attack_steps: int = 50
    n_samples: int = 200


class BatchEvalResult(BaseModel):
    model_id: str
    epsilons: list[float]
    accuracy: list[float]
    macro_dprime: list[float]
    pooled_dprime: list[float]
    thresh_macro: Optional[float] = None
    thresh_pooled: Optional[float] = None


class SaliencyRequest(BaseModel):
    model_id: str
    method: str = "gradcam"


class BenchmarkResult(BaseModel):
    model_id: str
    clean_acc: float
    rob_acc_at_eps: dict[str, float]
    ethresh: float
    params_million: float
    inference_ms: float
