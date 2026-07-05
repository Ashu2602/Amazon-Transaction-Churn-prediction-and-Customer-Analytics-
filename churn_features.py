"""
Build a churn-labeled, leakage-free feature table from the Amazon purchases + survey data.

Churn definition (rigor note):
Most users' last recorded purchase clusters tightly around Dec 2022 - Mar 2023 --
this is almost certainly when each person's data extraction/survey consent happened,
NOT organic churn. If we defined churn as "no purchase after this user's own last
date" every user would trivially churn (right-censoring tautology: there's no data
after the point where data collection stopped for them).

Instead we use a FIXED global time split so churn is measured against a common clock:
  - R1 (feature cutoff):  2022-06-01  -> everything on/before this builds features
  - R2 (label horizon):   2023-03-15  -> ~90th percentile of last-purchase dates, so
                            almost every user's data extends at least this far
  churn = 1 if a user has ZERO purchases in (R1, R2], else 0

Only "established" users are eligible (>=5 purchases before R1, tenure >=90 days
before R1) so we're not scoring brand-new accounts that never had a chance to churn.
"""
import pandas as pd

DATA_DIR = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data\Amazon"
OUT_PATH = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data\Code files\churn_dataset.csv"

R1 = pd.Timestamp("2022-06-01")  # feature cutoff -- nothing after this may be used as a feature
R2 = pd.Timestamp("2023-03-15")  # label horizon end
MIN_ORDERS_BEFORE_R1 = 5
MIN_TENURE_DAYS = 90

# Survey columns intentionally excluded on ethical grounds: substance use, health
# conditions, sexual orientation. Churn prediction shouldn't be built on those even
# if they're statistically predictive -- they're sensitive attributes unrelated to
# the business question and using them would be an easy way to bake in discrimination.
SAFE_SURVEY_COLS = [
    "Survey ResponseID",
    "Q-demos-age",
    "Q-demos-income",
    "Q-demos-education",
    "Q-amazon-use-hh-size",
    "Q-amazon-use-howmany",
    "Q-amazon-use-how-oft",
]


def load_purchases():
    df = pd.read_csv(
        f"{DATA_DIR}\\amazon-purchases.csv",
        parse_dates=["Order Date"],
        usecols=["Order Date", "Purchase Price Per Unit", "Quantity", "Category", "Survey ResponseID"],
    )
    df["spend"] = df["Purchase Price Per Unit"] * df["Quantity"]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df["Order Date"] <= R1]  # feature window -- strictly no data after R1
    label_window = df[(df["Order Date"] > R1) & (df["Order Date"] <= R2)]

    g = pre.groupby("Survey ResponseID")

    feats = pd.DataFrame(index=g.size().index)
    feats["n_orders"] = g.size()
    feats["first_order"] = g["Order Date"].min()
    feats["last_order"] = g["Order Date"].max()
    feats["tenure_days"] = (R1 - feats["first_order"]).dt.days
    feats["recency_days"] = (R1 - feats["last_order"]).dt.days  # days since last order, as of R1
    feats["total_spend"] = g["spend"].sum()
    feats["avg_order_value"] = g["spend"].mean()
    feats["n_categories"] = g["Category"].nunique()
    # avg gap between orders -- a low-frequency shopper looks different from a lapsing one
    feats["avg_days_between_orders"] = feats["tenure_days"] / feats["n_orders"].clip(lower=1)

    # momentum: orders in the 90 days right before R1 vs. the 90 days before that.
    # A shrinking recent count is a much stronger churn signal than raw frequency alone.
    last90 = pre[pre["Order Date"] > (R1 - pd.Timedelta(days=90))].groupby("Survey ResponseID").size()
    prev90 = pre[
        (pre["Order Date"] <= (R1 - pd.Timedelta(days=90)))
        & (pre["Order Date"] > (R1 - pd.Timedelta(days=180)))
    ].groupby("Survey ResponseID").size()
    feats["orders_last_90d"] = last90.reindex(feats.index).fillna(0)
    feats["orders_prior_90d"] = prev90.reindex(feats.index).fillna(0)
    feats["momentum"] = feats["orders_last_90d"] - feats["orders_prior_90d"]

    # eligibility gate -- established customers only, see module docstring
    eligible = feats[(feats["n_orders"] >= MIN_ORDERS_BEFORE_R1) & (feats["tenure_days"] >= MIN_TENURE_DAYS)].copy()

    # label computed ONLY from the (R1, R2] window -- never touches feature-window data
    future_counts = label_window.groupby("Survey ResponseID").size()
    eligible["orders_in_label_window"] = future_counts.reindex(eligible.index).fillna(0)
    eligible["churn"] = (eligible["orders_in_label_window"] == 0).astype(int)

    return eligible.drop(columns=["first_order", "last_order"])


def main():
    purchases = load_purchases()
    features = build_features(purchases)

    survey = pd.read_csv(f"{DATA_DIR}\\survey.csv", usecols=SAFE_SURVEY_COLS)
    dataset = features.merge(survey, left_index=True, right_on="Survey ResponseID", how="left").set_index(
        "Survey ResponseID"
    )

    print("Final dataset shape:", dataset.shape)
    print("Churn rate:", dataset["churn"].mean().round(4), f"({dataset['churn'].sum()} churned / {len(dataset)} total)")
    print("\nMissing values:\n", dataset.isna().sum())
    print("\nFeature preview:\n", dataset.head())

    dataset.to_csv(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
