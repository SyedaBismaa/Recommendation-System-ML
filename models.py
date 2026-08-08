import pandas as pd


df=pd.read_csv('./datasets/model_ready_ds.csv')

# print(df.shape)
# print(df.isna().sum())
# print(df.dtypes)



df = df.drop(columns=[
    "customer_since","account_type","monthly_savings","emi","credit_score","debt_to_income_ratio",
    "credit_card","fixed_deposit","recurring_deposit","mutual_fund","insurance","demat_account","forex_card",
    "travel_card","total_transactions","avg_amount","annual_income_log","average_balance_log","loan_amount_log","monthly_expenses_log",
    "has_loan","risk_profile_High","risk_profile_Low","risk_profile_Medium","savings_ratio","expense_ratio",
    "loan_burden","balance_to_income","product_count","loan_count","investment_count","digital_engagement"
])

print(df.columns)