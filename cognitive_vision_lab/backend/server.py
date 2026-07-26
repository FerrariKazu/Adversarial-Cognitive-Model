import io
import json
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, Response
from PIL import Image

from cognitive_vision_lab.config import (
    SWEEP_PATH,
    BACKEND_HOST,
    BACKEND_PORT,
    STL10_CLASSES,
)
from cognitive_vision_lab.backend.model_registry import (
    load_model,
    list_available_models,
    predict,
    load_imagenet_labels,
)
from cognitive_vision_lab.backend.attacks import pgd_attack
from cognitive_vision_lab.backend.explainability import GradCAM, extract_attention_maps, compute_representation
from cognitive_vision_lab.backend.schemas import (
    InferenceRequest,
    InferenceResult,
    SaliencyRequest,
)

app = FastAPI(title="Cognitive Vision Lab API", version="1.0.0")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models_available": len(list_available_models()),
    }


@app.get("/api/models")
def get_models():
    return {"models": list_available_models()}


@app.get("/api/sweep-results")
def get_sweep_results():
    if SWEEP_PATH.exists():
        with open(SWEEP_PATH) as f:
            data = json.load(f)
        return data
    return {}


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    model_id: str = Query(...),
    attack_eps: Optional[float] = Query(None),
    attack_steps: Optional[int] = Query(None),
):
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    model, transform = load_model(model_id)
    is_stl10 = model_id in [
        "RHAN-Large (ep45)", "RHAN-v10 (Final)", "RHAN-v11 (Best)",
    ]
    if not is_stl10:
        is_stl10 = False

    input_tensor = transform(pil_img).to(next(model.parameters()).device)

    result = predict(model, input_tensor, stl10=is_stl10)

    if attack_eps is not None and attack_eps > 0:
        adv_tensor = pgd_attack(
            model,
            input_tensor,
            label_idx=result["predicted_idx"],
            eps=attack_eps,
            steps=attack_steps or 40,
        )
        adv_result = predict(model, adv_tensor, stl10=is_stl10)
        result["adversarial"] = adv_result
        result["attack_eps"] = attack_eps
        result["attack_steps"] = attack_steps or 40

    return result


@app.post("/api/saliency")
async def api_saliency(
    file: UploadFile = File(...),
    model_id: str = Query(...),
    method: str = Query("gradcam"),
):
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    model, transform = load_model(model_id)
    input_tensor = transform(pil_img).to(next(model.parameters()).device)

    cam = GradCAM(model)
    heatmap = cam.generate(input_tensor)

    import numpy as np
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    from PIL import Image as PILImage
    heatmap_img = PILImage.fromarray(heatmap_uint8, mode="L")

    buf = io.BytesIO()
    heatmap_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/attention")
async def api_attention(file: UploadFile = File(...), model_id: str = Query(...)):
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    model, transform = load_model(model_id)
    input_tensor = transform(pil_img).to(next(model.parameters()).device)

    attn_maps = extract_attention_maps(model, input_tensor)
    result = {}
    for name, attn in attn_maps.items():
        result[name] = {
            "shape": list(attn.shape),
            "mean": float(attn.mean().item()),
            "std": float(attn.std().item()),
        }
    return result


@app.post("/api/representation")
async def api_representation(file: UploadFile = File(...), model_id: str = Query(...)):
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    model, transform = load_model(model_id)
    input_tensor = transform(pil_img).to(next(model.parameters()).device)

    reps = compute_representation(model, input_tensor)
    result = {}
    for name, rep in reps.items():
        result[name] = {
            "shape": list(rep.shape),
            "mean": float(rep.mean().item()),
            "std": float(rep.std().item()),
            "norm": float(rep.norm().item()),
        }
    return result


def serve():
    import uvicorn
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)


if __name__ == "__main__":
    serve()
