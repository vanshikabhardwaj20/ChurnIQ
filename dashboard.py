
import pandas as pd

# Load the original customer dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

# Calculate overall customer statistics
total_customers = len(df)
churned_customers = int((df["Churn"] == "Yes").sum())
churn_rate = round(
    (churned_customers / total_customers) * 100, 2
)

print("\n" + "=" * 45)
print("        CHURNIQ ANALYTICS")
print("=" * 45)

print(f"Total customers : {total_customers}")
print(f"Churned customers: {churned_customers}")
print(f"Customers retained: {total_customers - churned_customers}")
print(f"Overall churn rate: {churn_rate}%")

# Calculate churn rate by contract type
print("\nCHURN RATE BY CONTRACT TYPE")
print("-" * 45)

contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
    .sort_values(ascending=False)
)

for contract, rate in contract_churn.items():
    print(f"{contract:20} {rate:6.2f}%")

# Calculate churn rate by tenure group
print("\nCHURN RATE BY CUSTOMER TENURE")
print("-" * 45)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months",
            "25-48 months", "49-72 months"]
)

tenure_churn = (
    df.groupby("TenureGroup", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
)

for group, rate in tenure_churn.items():
    print(f"{str(group):20} {rate:6.2f}%")

print("\nAnalytics calculation complete!")
