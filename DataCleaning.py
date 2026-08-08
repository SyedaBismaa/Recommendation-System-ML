import pandas as pd
import numpy as np

df = pd.read_csv("./datasets/ml_dataset.csv")

# print(df.shape)
# print(df.columns)
# print(df.isna().sum)
# print(df.dtype)

df['age'] = df['age'].fillna(df['age'].median())
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
df['marital_status'] = df['marital_status'].fillna(df['marital_status'].mode()[0])
df['education'] = df['education'].fillna(df['education'].mode()[0])
df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
df['state'] = df['state'].fillna(df['state'].mode()[0])
df['employment_status'] = df['employment_status'].fillna(df['employment_status'].mode()[0])
df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())
df['monthly_income'] = df['monthly_income'].fillna(df['monthly_income'].median())
df['customer_since'] = df['customer_since'].fillna(df['customer_since'].median())
df['customer_segment'] = df['customer_segment'].fillna(df['customer_segment'].mode()[0])
df['risk_profile'] = df['risk_profile'].fillna(df['risk_profile'].mode()[0])
df['account_type'] = df['account_type'].fillna(df['account_type'].mode()[0])
df['average_balance'] = df['average_balance'].fillna(df['average_balance'].median())
df['monthly_expenses'] = df['monthly_expenses'].fillna(df['monthly_expenses'].median())
df['monthly_savings'] = df['monthly_savings'].fillna(df['monthly_savings'].median())
df['emi'] = df['emi'].fillna(df['emi'].median())
df['loan_amount'] = df['loan_amount'].fillna(df['loan_amount'].median())
df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
df['debt_to_income_ratio'] = df['debt_to_income_ratio'].fillna(df['debt_to_income_ratio'].median())
df['number_of_accounts'] = df['number_of_accounts'].fillna(df['number_of_accounts'].median())
df['digital_banking_usage'] = df['digital_banking_usage'].fillna(df['digital_banking_usage'].median())
df['mobile_app_usage'] = df['mobile_app_usage'].fillna(df['mobile_app_usage'].median())
df['atm_transactions'] = df['atm_transactions'].fillna(df['atm_transactions'].median())
df['online_transactions'] = df['online_transactions'].fillna(df['online_transactions'].median())

#  Outlier tratment using log transform
skewed_cols = ['annual_income','monthly_income','monthly_expenses','average_balance','loan_amount','total_amount']
for col in skewed_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

# logTransform
for col in skewed_cols:
    df[col + '_log'] = np.log1p(df[col])

df['has_loan'] = (df['loan_amount'] > 0).astype(int)

#one hot
df = pd.get_dummies(df, columns=['gender','marital_status','education','occupation','state',
                                  'employment_status','risk_profile'], dtype=int)

# Bining
def age_group(age):
    if 18 <= age <= 25: 
        return "Young"
    elif 26 <= age <= 40:
        return "Adult"
    elif 41 <= age <= 60:
        return "Middle_Age"
    else:
        return "Senior"

def customer_group(cus_since):
    if 0 <= cus_since <= 3: 
        return "New"
    elif 4 <= cus_since <= 8: 
        return "Regular"
    elif 9 <= cus_since <= 15: 
        return "Loyal"
    else:
        return "Veteran"

df["age_group"] = df["age"].apply(age_group)
df["customer_since_group"] = df["customer_since"].apply(customer_group)
df = pd.get_dummies(df, columns=['age_group','customer_since_group'], dtype=int)

#Feature Engineering 
df['savings_ratio'] = df['monthly_savings'] / df['monthly_income']
df['expense_ratio'] = df['monthly_expenses'] / df['monthly_income']
df['loan_burden'] = df['loan_amount'] / df['annual_income']
df['balance_to_income'] = df['average_balance'] / df['annual_income']

product_cols = ['salary_account','savings_account','credit_card','personal_loan','home_loan','car_loan','gold_loan',
                'fixed_deposit','recurring_deposit','mutual_fund','insurance','demat_account','forex_card','travel_card']
df['product_count'] = df[product_cols].sum(axis=1)

loan_cols = ['personal_loan','home_loan','car_loan','gold_loan']
df['loan_count'] = df[loan_cols].sum(axis=1)

investment_cols = ['fixed_deposit','recurring_deposit','mutual_fund','insurance','demat_account']
df['investment_count'] = df[investment_cols].sum(axis=1)

df['digital_engagement'] = (df['mobile_app_usage'] + df['digital_banking_usage']) / 2


df = df.drop(columns=['monthly_income'])

df.to_csv('./datasets/model_ready_ds.csv', index=False)

print(df.shape)
print(df.isna().sum().sum())