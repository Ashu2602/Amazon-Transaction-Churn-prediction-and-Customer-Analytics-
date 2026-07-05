"""Manual smoke test for the retrieval/recommendation logic -- no Claude/API
needed, since this exercises recommend_tools.py directly. Run after
build_product_index.py finishes."""
import pandas as pd
from recommend_tools import get_index

pd.set_option("display.max_colwidth", 60)


def main():
    idx = get_index()

    print("=== search_products('wireless bluetooth headphones') ===")
    print(idx.search_products("wireless bluetooth headphones", k=5)[["title", "category", "avg_price", "similarity"]])

    sample_user = idx.purchases["Survey ResponseID"].iloc[0]
    print(f"\n=== get_user_history({sample_user}) ===")
    hist = idx.get_user_history(sample_user, n=5)
    print(hist[["Order Date", "Title", "Category"]])

    print(f"\n=== recommend_for_user({sample_user}) ===")
    try:
        recs = idx.recommend_for_user(sample_user, k=5)
        print(recs[["title", "category", "avg_price", "similarity"]])
    except ValueError as e:
        print("skipped:", e)

    print("\n=== category_stats('PET_FOOD') ===")
    print(idx.category_stats("PET_FOOD"))


if __name__ == "__main__":
    main()
