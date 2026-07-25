"""QuantFlow ML Inference Engine — PyTorch (CPU-optimized)."""
import json, logging, os, time, torch, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "checkpoints/cascade_anp.pt"))
DATA_PATH = Path(os.environ.get("DATA_PATH", "data/alpaca_5m.parquet"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "data/cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

model = None
feature_cols = []
ticker_cache = {}

class PredictRequest(BaseModel):
    ticker: str; bars: Optional[List[List[float]]] = None; n_context: int = 80
class PredictResponse(BaseModel):
    ticker: str; predicted_return: float; aleatoric_sigma: float
    direction: str; confidence: float; inference_ms: float
class BatchRequest(BaseModel):
    tickers: List[str]; n_context: int = 80
class BatchResponse(BaseModel):
    predictions: List[PredictResponse]; total_ms: float

def get_feature_cols(df):
    exclude = {"ticker","event_state","event_state_code","is_volatile","tau_forward",
        "target_return","target_5d","target_21d","target_1h","target_4h",
        "as_of_date","adj_close","close","drawdown_5d_max","drawdown_21d_max",
        "target_direction_5d","target_direction_21d","sector_idx","target_return_path"}
    return sorted([c for c in df.columns if c not in exclude and df[c].dtype!='object'])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_cols
    from quantflow.models.cascade_anp import CascadeANP
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    df = pd.read_parquet(DATA_PATH)
    feature_cols = get_feature_cols(df)
    model = CascadeANP(len(feature_cols), cfg["hidden_dim"], cfg.get("latent_dim",64),
                       cfg.get("cascade_layers",4))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params, "
                f"epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss',0):.4f}")
    yield

app = FastAPI(title="QuantFlow ML Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/models")
async def list_models():
    import json
    registry_path = Path(os.environ.get("MODEL_REGISTRY", "checkpoints/model_registry.json"))
    if registry_path.exists():
        with open(registry_path) as f:
            return json.load(f)
    return {"models": [], "updated_at": None}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None,
            "features": len(feature_cols), "cached_tickers": len(ticker_cache)}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None: raise HTTPException(503, "Model not loaded")
    t0 = time.time()
    data = ticker_cache.get(req.ticker)
    if data is None and DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
        g = df[df.ticker == req.ticker].sort_values("as_of_date")
        if len(g) < req.n_context + 22:
            raise HTTPException(400, f"Not enough data for {req.ticker}")
        data = {"feats": g[feature_cols].fillna(0).values.astype(np.float32),
                "returns": g["target_return"].fillna(0).values.astype(np.float32)}
        ticker_cache[req.ticker] = data

    feats = data["feats"]; returns = data["returns"]
    bar = min(len(feats) - 22, 700); ctx_s = bar - req.n_context

    # Get cascade-encoded context tokens
    rf_window = torch.from_numpy(feats[ctx_s:bar]).unsqueeze(0)
    with torch.no_grad():
        fused, _, _ = model.encode(rf_window)
    xc = fused[:, :, :]  # [1, N, 128] cascade tokens
    yc = torch.from_numpy(returns[ctx_s:bar,None]).unsqueeze(0)

    with torch.no_grad():
        out = model(xc, yc, n_steps=21, teacher_forcing=False, raw_features=rf_window)
    pred_ret = float(out["predicted_return"])
    sigma = float(out["aleatoric_sigma"])
    direction = "BUY" if pred_ret > 0.002 else "SELL" if pred_ret < -0.002 else "HOLD"
    confidence = 1.0 / (1.0 + 2.0 * sigma)

    return PredictResponse(ticker=req.ticker, predicted_return=pred_ret,
                           aleatoric_sigma=sigma, direction=direction,
                           confidence=confidence, inference_ms=(time.time()-t0)*1000)

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
    t0 = time.time(); preds = []
    for t in req.tickers[:32]:
        try: preds.append(await predict(PredictRequest(ticker=t, n_context=req.n_context)))
        except Exception as e: logger.warning(f"Skipping {t}: {e}")
    return BatchResponse(predictions=preds, total_ms=(time.time()-t0)*1000)

_start = time.time()
