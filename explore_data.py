"""Initial exploration of the Amazon purchases + survey dataset."""
import pandas as pd

DATA_DIR = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data\Amazon"

purchases = pd.read_csv(f"{DATA_DIR}\\amazon-purchases.csv", parse_dates=["Order Date"])
survey = pd.read_csv(f"{DATA_DIR}\\survey.csv")

print("=== PURCHASES ===")
print("shape:", purchases.shape)
print(purchases.dtypes)
print("\nDate range:", purchases["Order Date"].min(), "to", purchases["Order Date"].max())
print("\nUnique users (Survey ResponseID):", purchases["Survey ResponseID"].nunique())
print("\nMissing values:\n", purchases.isna().sum())
print("\nNum unique categories:", purchases["Category"].nunique())
print("\nTop 20 categories:\n", purchases["Category"].value_counts().head(20))
print("\nPrice stats:\n", purchases["Purchase Price Per Unit"].describe())
print("\nQuantity stats:\n", purchases["Quantity"].describe())
print("\nTop states:\n", purchases["Shipping Address State"].value_counts().head(10))

print("\n\n=== SURVEY ===")
print("shape:", survey.shape)
print("\nUnique response IDs:", survey["Survey ResponseID"].nunique())
print("\nColumns:", list(survey.columns))

print("\n\n=== LINKAGE ===")
common_ids = set(purchases["Survey ResponseID"]).intersection(set(survey["Survey ResponseID"]))
print("Users present in both purchases and survey:", len(common_ids))
print("Users in purchases only:", len(set(purchases["Survey ResponseID"]) - set(survey["Survey ResponseID"])))
print("Users in survey only:", len(set(survey["Survey ResponseID"]) - set(purchases["Survey ResponseID"])))
