from __future__ import annotations

import re
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="LLM Shield Prototype",
    version="1.1.0",
    description="Prototype adversarial prompt detector based on TF-IDF + Linear SVM.",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Demo-only in-memory storage.
EVENTS: deque[dict[str, Any]] = deque(maxlen=200)
PROMPT_HISTORY: deque[str] = deque(maxlen=500)
MODEL_BUNDLE: dict[str, Any] | None = None

OVERRIDE_PATTERNS = [
    r"ignore previous instructions",
    r"disregard .* instructions",
    r"bypass",
    r"uncensored",
    r"developer mode",
    r"override",
    r"jailbreak",
    r"forget all prior rules",
    r"ignore the system prompt",
    r"act as an unrestricted",
]

SUSPICIOUS_TERMS = {
    "ignore",
    "bypass",
    "override",
    "uncensored",
    "hack",
    "disable",
    "exploit",
    "reveal",
    "forget",
    "jailbreak",
    "evade",
}


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20000)
    source: str = Field(default="manual-ui", max_length=200)



def load_model(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise RuntimeError(
            "Model file not found. Run `python train.py --data sample_prompts.csv --model-out model.joblib` first."
        )
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise RuntimeError("Loaded model bundle is invalid.")
    return bundle


@app.on_event("startup")
def startup_event() -> None:
    global MODEL_BUNDLE
    MODEL_BUNDLE = load_model()



def ensure_model_loaded() -> dict[str, Any]:
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        try:
            MODEL_BUNDLE = load_model()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MODEL_BUNDLE



def sigmoid_like_from_margin(margin: float) -> float:
    # LinearSVC exposes a signed distance rather than calibrated probabilities.
    # This score is only for UI intuition and must not be treated as a true probability.
    return max(0.0, min(1.0, 1 / (1 + pow(2.718281828, -margin))))



def analyze_prompt_text(prompt: str) -> dict[str, Any]:
    prompt_lower = prompt.lower()
    detected_patterns = [pattern for pattern in OVERRIDE_PATTERNS if re.search(pattern, prompt_lower)]

    words = re.findall(r"\b\w+\b", prompt_lower)
    total_words = len(words) or 1
    unique_words = len(set(words))
    lexical_diversity = round(unique_words / total_words, 3)
    suspicious_terms = [word for word in words if word in SUSPICIOUS_TERMS]
    sentence_split = [s.strip() for s in re.split(r"[.!?]+", prompt) if s.strip()]
    avg_words_per_sentence = round(total_words / max(len(sentence_split), 1), 2)

    return {
        "word_count": total_words,
        "sentence_count": len(sentence_split),
        "avg_words_per_sentence": avg_words_per_sentence,
        "lexical_diversity": lexical_diversity,
        "override_patterns": detected_patterns,
        "suspicious_terms": Counter(suspicious_terms).most_common(10),
    }



def predict_prompt(prompt: str, source: str) -> dict[str, Any]:
    bundle = ensure_model_loaded()
    model = bundle["model"]

    prediction = int(model.predict([prompt])[0])
    margin = float(model.decision_function([prompt])[0])
    confidence = round(sigmoid_like_from_margin(abs(margin)), 4)
    label_name = bundle.get("labels", {}).get(prediction, str(prediction))
    analysis = analyze_prompt_text(prompt)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "prompt": prompt,
        "label": prediction,
        "label_name": label_name,
        "confidence": confidence,
        "analysis": analysis,
    }
    EVENTS.appendleft(event)
    PROMPT_HISTORY.append(prompt)
    return event



def build_dashboard_stats() -> dict[str, Any]:
    bundle = MODEL_BUNDLE or {}
    metrics = bundle.get("metrics", {})
    total = len(EVENTS)
    safe_count = sum(1 for event in EVENTS if event["label"] == 0)
    adversarial_count = sum(1 for event in EVENTS if event["label"] == 1)
    source_counts = Counter(event["source"] for event in EVENTS).most_common(5)

    return {
        "total_scans": total,
        "safe_count": safe_count,
        "adversarial_count": adversarial_count,
        "recent_sources": source_counts,
        "training_accuracy": round(metrics.get("accuracy", 0.0), 4),
        "train_size": metrics.get("train_size", 0),
        "test_size": metrics.get("test_size", 0),
        "dataset_name": metrics.get("dataset_name", "unknown"),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": build_dashboard_stats(),
            "events": list(EVENTS)[:12],
            "metrics": (MODEL_BUNDLE or {}).get("metrics", {}),
        },
    )


@app.get("/health")
def health() -> dict[str, Any]:
    model_ready = MODEL_BUNDLE is not None or MODEL_PATH.exists()
    return {"status": "ok" if model_ready else "degraded", "model_loaded": model_ready}


@app.get("/stats")
def stats() -> dict[str, Any]:
    return build_dashboard_stats()


@app.get("/events")
def events() -> dict[str, Any]:
    return {"events": list(EVENTS)}


@app.post("/predict")
def predict(payload: PromptRequest) -> JSONResponse:
    result = predict_prompt(prompt=payload.prompt.strip(), source=payload.source.strip() or "api")
    return JSONResponse(result)


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported in this prototype.")

    contents = await file.read()
    text = contents.decode("utf-8", errors="ignore").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = predict_prompt(text, source=f"file:{file.filename}")
    return JSONResponse(result)
