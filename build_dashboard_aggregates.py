"""
Precompute small aggregate tables for the Streamlit dashboard.

Why precompute instead of loading amazon-purchases.csv directly in the app:
the raw file is 1.85M rows / ~300MB, and pandas typically uses 3-5x a CSV's
size in memory once loaded -- comfortably blowing past the ~1GB RAM budget on
Streamlit Community Cloud's free tier. These aggregates are all the dashboard
actually needs (category/state/monthly rollups), and together they're a few
hundred KB, so the hosted app loads in-memory data only, never the raw file.
"""
import pandas as pd

DATA_DIR = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data\Amazon"
OUT_DIR = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data\Code files\dashboard_data"

import os
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(
        f"{DATA_DIR}\\amazon-purchases.csv",
        parse_dates=["Order Date"],
        usecols=["Order Date", "Purchase Price Per Unit", "Quantity", "Category", "Shipping Address State"],
    )
    df["spend"] = df["Purchase Price Per Unit"] * df["Quantity"]

    category = (
        df.dropna(subset=["Category"])
        .groupby("Category")
        .agg(n_purchases=("spend", "size"), total_spend=("spend", "sum"), avg_price=("Purchase Price Per Unit", "mean"))
        .sort_values("total_spend", ascending=False)
        .reset_index()
    )
    category.to_csv(f"{OUT_DIR}\\category_stats.csv", index=False)

    state = (
        df.dropna(subset=["Shipping Address State"])
        .groupby("Shipping Address State")
        .agg(n_purchases=("spend", "size"), total_spend=("spend", "sum"), avg_price=("Purchase Price Per Unit", "mean"))
        .sort_values("total_spend", ascending=False)
        .reset_index()
    )
    state.to_csv(f"{OUT_DIR}\\state_stats.csv", index=False)

    monthly = df.copy()
    monthly["year_month"] = monthly["Order Date"].dt.to_period("M").astype(str)
    monthly = (
        monthly.groupby("year_month")
        .agg(n_orders=("spend", "size"), total_spend=("spend", "sum"))
        .reset_index()
        .sort_values("year_month")
    )
    monthly.to_csv(f"{OUT_DIR}\\monthly_trend.csv", index=False)

    print("category_stats:", category.shape)
    print("state_stats:", state.shape)
    print("monthly_trend:", monthly.shape)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
