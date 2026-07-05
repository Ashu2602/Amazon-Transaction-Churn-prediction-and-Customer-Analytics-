"""
Loads the small precomputed aggregate tables (see build_dashboard_aggregates.py)
plus the churn dataset. Kept separate from recommend_tools.py deliberately --
this dashboard never touches the raw 1.85M-row purchases file or the product
embedding index, so the hosted app's memory footprint stays tiny.
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")


def load_tables() -> dict[str, pd.DataFrame]:
    category = pd.read_csv(os.path.join(DASHBOARD_DATA_DIR, "category_stats.csv"))
    state = pd.read_csv(os.path.join(DASHBOARD_DATA_DIR, "state_stats.csv"))
    monthly = pd.read_csv(os.path.join(DASHBOARD_DATA_DIR, "monthly_trend.csv"))
    churn = pd.read_csv(os.path.join(BASE_DIR, "churn_dataset.csv"))
    return {
        "category": category,
        "state": state,
        "monthly": monthly,
        "churn_users": churn,
    }


TABLE_DESCRIPTIONS = """
Available tables:

- "category": one row per product category (1,871 rows). Columns: Category,
  n_purchases, total_spend, avg_price.
- "state": one row per US state, 2-letter code (52 rows). Columns:
  Shipping Address State, n_purchases, total_spend, avg_price.
- "monthly": one row per calendar month, 2018-01 through ~2023-03 (68 rows).
  Columns: year_month (YYYY-MM string), n_orders, total_spend.
- "churn_users": one row per surveyed customer (4,930 rows) -- the labeled
  churn dataset. Columns: n_orders, tenure_days, recency_days, total_spend,
  avg_order_value, n_categories, avg_days_between_orders, orders_last_90d,
  orders_prior_90d, momentum, churn (0=retained, 1=churned), Q-demos-age,
  Q-demos-education, Q-demos-income, Q-amazon-use-howmany,
  Q-amazon-use-hh-size, Q-amazon-use-how-oft.
"""
