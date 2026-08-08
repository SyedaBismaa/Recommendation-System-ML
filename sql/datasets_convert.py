import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

print("✅ Connected Successfully!")



df1 = pd.read_csv("datasets/customers.csv")
df1.columns = df1.columns.str.lower()

# df1.to_sql(
#     "customers",
#     engine,
#     if_exists="append",
#     index=False
# )



df2 = pd.read_csv("datasets/accounts.csv")
df2.columns = df2.columns.str.lower()

# df2.to_sql(
#     "accounts",
#     engine,
#     if_exists="append",
#     index=False
# )


df3 = pd.read_csv("datasets/transactions.csv")
df3.columns = df3.columns.str.lower()

# df3.to_sql(
#     "transactions",
#     engine,
#     if_exists="append",
#     index=False
# )



df4 = pd.read_csv("datasets/products.csv")
df4.columns = df4.columns.str.lower()

bool_cols = [
    "salary_account","savings_account","credit_card",
    "personal_loan","home_loan","car_loan","gold_loan",
    "fixed_deposit","recurring_deposit","mutual_fund",
    "insurance","demat_account", "forex_card","travel_card"
]

df4[bool_cols] = df4[bool_cols].astype(bool)

# df4.to_sql(
#     "banking_products",
#     engine,
#     if_exists="append",
#     index=False
# )


print("✅ DataFrames Ready!")