"""
FastAPI backend for the Lovable frontend. Wraps three things behind REST:
  1. Churn risk lookup (precomputed model, churn_model.joblib)
  2. Product search / similarity / personalized recommendations (recommend_tools.py)
  3. Conversational agent (agent.py, Claude Agent SDK -- needs ANTHROPIC_API_KEY)

CORS is wide open (allow_origins=["*"]) because Lovable's preview environment
runs on a dynamic origin; tighten this to the deployed frontend's real origin
before this goes anywhere near production, since "*" plus credentials would be
a real cross-origin risk (we don't send credentials here, so it's contained,
but it's still not a setting to carry forward blindly).
"""
import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from recommend_tools import get_index

load_dotenv()  # loads ANTHROPIC_API_KEY from .env (gitignored) for the /chat endpoint

CODE_DIR = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data\Code files"

app = FastAPI(title="Amazon Purchases Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_churn_model = joblib.load(f"{CODE_DIR}\\churn_model.joblib")
_churn_dataset = pd.read_csv(f"{CODE_DIR}\\churn_dataset.csv", index_col="Survey ResponseID")
_FEATURE_COLS = [c for c in _churn_dataset.columns if c not in ("churn", "orders_in_label_window")]


class ChatRequest(BaseModel):
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/users/{response_id}/churn-risk")
def churn_risk(response_id: str):
    if response_id not in _churn_dataset.index:
        raise HTTPException(404, f"{response_id} not in the scored churn population (needs >=5 orders, >=90d tenure before 2022-06-01)")
    row = _churn_dataset.loc[[response_id], _FEATURE_COLS]
    proba = _churn_model.predict_proba(row)[0, 1]
    return {"response_id": response_id, "churn_probability": round(float(proba), 4), "actual_label": int(_churn_dataset.loc[response_id, "churn"])}


@app.get("/users/{response_id}/history")
def user_history(response_id: str, n: int = 20):
    idx = get_index()
    hist = idx.get_user_history(response_id, n=n)
    if hist.empty:
        raise HTTPException(404, f"No purchase history for {response_id}")
    return hist.to_dict(orient="records")


@app.get("/users/{response_id}/recommendations")
def recommendations(response_id: str, k: int = 10):
    idx = get_index()
    try:
        recs = idx.recommend_for_user(response_id, k=k)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return recs.to_dict(orient="records")


@app.get("/search")
def search(q: str, k: int = 10):
    idx = get_index()
    return idx.search_products(q, k=k).to_dict(orient="records")


@app.get("/products/{asin}/similar")
def similar_products(asin: str, k: int = 10):
    idx = get_index()
    try:
        return idx.similar_to_asin(asin, k=k).to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    from agent import ask  # imported lazily -- only needed if this endpoint is actually hit

    try:
        reply = await ask(req.prompt)
    except Exception as e:
        raise HTTPException(503, f"Agent unavailable (is ANTHROPIC_API_KEY set?): {e}")
    return {"reply": reply}
