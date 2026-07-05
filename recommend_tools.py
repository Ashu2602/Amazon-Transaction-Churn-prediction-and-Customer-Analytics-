"""
Retrieval + content-based recommendation over the Amazon product embedding index.

This module is framework-agnostic on purpose: both the Claude Agent SDK tool
wrappers (agent.py) and the FastAPI backend (api.py) import these same
functions, so the retrieval logic only has to be written -- and tested -- once.

Recommendation approach: content-based, not collaborative filtering. We don't
have enough per-user purchase overlap to do item-item collaborative filtering
well (each user's history is a tiny slice of an 876K-product catalog), so
instead we build a user's "taste vector" as the purchase-weighted centroid of
the embeddings of things they've already bought, then retrieve nearby products
they haven't bought yet. This is the standard fallback when the interaction
matrix is too sparse for CF to have enough signal.
"""
import numpy as np
import pandas as pd
from functools import lru_cache
from sentence_transformers import SentenceTransformer

BASE_DIR = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data"
CODE_DIR = f"{BASE_DIR}\\Code files"
DATA_DIR = f"{BASE_DIR}\\Amazon"
MODEL_NAME = "all-MiniLM-L6-v2"


class ProductIndex:
    """Holds embeddings + metadata + the raw purchases table in memory.

    Built once per process (FastAPI startup, or a single agent session) since
    loading a 1.85M-row CSV and an 876K x 384 float array on every call would
    make every tool invocation slow.
    """

    def __init__(self):
        self.embeddings = np.load(f"{CODE_DIR}\\product_embeddings.npy")  # (n_products, 384), L2-normalized
        self.metadata = pd.read_parquet(f"{CODE_DIR}\\product_metadata.parquet")
        self.asin_to_row = {asin: i for i, asin in enumerate(self.metadata["asin"])}
        self.model = SentenceTransformer(MODEL_NAME)

        self.purchases = pd.read_csv(
            f"{DATA_DIR}\\amazon-purchases.csv",
            parse_dates=["Order Date"],
            usecols=[
                "Order Date", "Purchase Price Per Unit", "Quantity",
                "Title", "ASIN/ISBN (Product Code)", "Category", "Survey ResponseID",
            ],
        ).rename(columns={"ASIN/ISBN (Product Code)": "asin"})

    def _top_k(self, query_vec: np.ndarray, k: int, exclude_asins: set[str] | None = None) -> pd.DataFrame:
        scores = self.embeddings @ query_vec  # cosine similarity (both sides pre-normalized)
        if exclude_asins:
            mask = self.metadata["asin"].isin(exclude_asins).to_numpy()
            scores = np.where(mask, -np.inf, scores)
        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        out = self.metadata.iloc[top_idx].copy()
        out["similarity"] = scores[top_idx]
        return out.reset_index(drop=True)

    def search_products(self, query: str, k: int = 10) -> pd.DataFrame:
        """Semantic search: free-text query -> top-k similar products."""
        query_vec = self.model.encode(query, normalize_embeddings=True)
        return self._top_k(query_vec, k)

    def similar_to_asin(self, asin: str, k: int = 10) -> pd.DataFrame:
        """Nearest neighbors to a specific product, e.g. 'what else is like this?'"""
        if asin not in self.asin_to_row:
            raise ValueError(f"Unknown ASIN: {asin}")
        query_vec = self.embeddings[self.asin_to_row[asin]]
        return self._top_k(query_vec, k, exclude_asins={asin})

    def get_user_history(self, response_id: str, n: int = 20) -> pd.DataFrame:
        """Most recent n purchases for a given survey respondent."""
        hist = self.purchases[self.purchases["Survey ResponseID"] == response_id]
        return hist.sort_values("Order Date", ascending=False).head(n)

    def recommend_for_user(self, response_id: str, k: int = 10, half_life_days: float = 180) -> pd.DataFrame:
        """Content-based recommendation: recency-weighted centroid of a user's
        purchased-product embeddings, then nearest unseen products.

        Recency weighting (exponential half-life) matters here because tastes
        drift over the multi-year span of this dataset -- a purchase from 2018
        shouldn't count as much as one from last month.
        """
        hist = self.purchases[self.purchases["Survey ResponseID"] == response_id].dropna(subset=["asin"])
        hist = hist[hist["asin"].isin(self.asin_to_row)]
        if hist.empty:
            raise ValueError(f"No indexed purchase history for {response_id}")

        rows = hist["asin"].map(self.asin_to_row).to_numpy()
        vecs = self.embeddings[rows]

        age_days = (hist["Order Date"].max() - hist["Order Date"]).dt.days.to_numpy()
        weights = 0.5 ** (age_days / half_life_days)
        weights = weights / weights.sum()

        taste_vec = (vecs * weights[:, None]).sum(axis=0)
        taste_vec = taste_vec / np.linalg.norm(taste_vec)

        already_bought = set(hist["asin"])
        return self._top_k(taste_vec, k, exclude_asins=already_bought)

    def category_stats(self, category: str) -> dict:
        rows = self.metadata[self.metadata["category"] == category]
        if rows.empty:
            raise ValueError(f"Unknown category: {category}")
        return {
            "category": category,
            "n_unique_products": len(rows),
            "total_purchases": int(rows["n_purchases"].sum()),
            "avg_price": float(rows["avg_price"].mean()),
        }


@lru_cache(maxsize=1)
def get_index() -> "ProductIndex":
    """Process-wide singleton so the embedding matrix / CSV load only happen once."""
    return ProductIndex()
