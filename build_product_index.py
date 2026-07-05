"""
Build a local semantic embedding index over unique Amazon products, used by the
retrieval/recommendation agent tools.

Why local embeddings instead of an API: we have ~876K unique (ASIN, Title)
products. Embedding that many items through a paid API on every rebuild is
wasteful and adds a hard dependency for something that only needs to happen
once, offline. sentence-transformers' all-MiniLM-L6-v2 is small (~90MB), fast
on CPU, and good enough for nearest-neighbor product similarity -- it doesn't
need to be state-of-the-art, just consistent.

We embed "<Title> [<Category>]" so category acts as a light disambiguator for
short/ambiguous titles, then store vectors + metadata in a single .npy/.parquet
pair for fast loading by the agent at query time.

Two deliberate scope cuts to keep the one-time build tractable on a laptop CPU:
  - Only products with >=2 historical purchases are indexed (876K -> ~219K).
    Singleton one-off purchases carry the weakest similarity signal anyway --
    there's nothing to validate "people who bought X also bought Y" against --
    so dropping them trades away the least useful part of the long tail.
  - Text is capped at 64 tokens (model.max_seq_length). A few outlier titles
    run past 1,000 words; transformer attention cost grows with sequence
    length, so a handful of pathological titles were dominating wall-clock
    time. 64 tokens comfortably covers the 95th percentile title (~32 words)
    and the product name/brand signal that matters for similarity lives in
    the first few words anyway.
"""
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data\Amazon"
OUT_DIR = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data\Code files"
MODEL_NAME = "all-MiniLM-L6-v2"


def clean_text(title: str, category: str) -> str:
    title = re.sub(r"\s+", " ", str(title)).strip()
    if pd.isna(category) or not category:
        return title
    return f"{title} [{category}]"


def main():
    print("Loading purchases...")
    df = pd.read_csv(
        f"{DATA_DIR}\\amazon-purchases.csv",
        usecols=["ASIN/ISBN (Product Code)", "Title", "Category", "Purchase Price Per Unit"],
    )
    df = df.rename(columns={"ASIN/ISBN (Product Code)": "asin"})
    df = df.dropna(subset=["Title"])

    # one row per product: keep the most recent price seen and the purchase
    # count as a popularity signal (both useful for the recommend tool later)
    agg = (
        df.groupby("asin")
        .agg(
            title=("Title", "first"),
            category=("Category", "first"),
            avg_price=("Purchase Price Per Unit", "mean"),
            n_purchases=("asin", "size"),
        )
        .reset_index()
    )
    print(f"Unique products (pre-filter): {len(agg):,}")
    agg = agg[agg["n_purchases"] >= 2].reset_index(drop=True)
    print(f"Unique products (n_purchases >= 2): {len(agg):,}")

    texts = [clean_text(t, c) for t, c in zip(agg["title"], agg["category"])]

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 64  # bounds attention cost; see module docstring

    print("Encoding (this is the one-time cost)...")
    embeddings = model.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity == dot product at query time
    )

    np.save(f"{OUT_DIR}\\product_embeddings.npy", embeddings.astype(np.float32))
    agg.to_parquet(f"{OUT_DIR}\\product_metadata.parquet", index=False)

    print(f"Saved {embeddings.shape[0]:,} vectors of dim {embeddings.shape[1]} to product_embeddings.npy")
    print("Saved metadata to product_metadata.parquet")


if __name__ == "__main__":
    main()
