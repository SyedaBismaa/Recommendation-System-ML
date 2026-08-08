import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv("./datasets/ml_dataset.csv")

print(df.columns)

# #univariant

num_cols=[
    'annual_income',
    'monthly_income',
    'average_balance',
    'loan_amount',
    'credit_score',
    'total_amount',
    'monthly_expenses',
]

# for col in num_cols:
#     plt.figure()
#     sns.histplot(df[col], kde=True)
#     plt.title(col)
#     plt.show()

# for col in num_cols:
#     plt.figure()
#     sns.boxplot(x=df[col])
#     plt.title(col)
#     plt.show()

#Plots have shown the distribution of numerical columns and identified outliers in the dataset.expect credit score 
#thus  using log transform and percentile capping

for col in ['annual_income','monthly_income', 'monthly_expenses','average_balance','loan_amount','total_amount']:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

for col in ['annual_income','monthly_income', 'monthly_expenses','average_balance','loan_amount','total_amount']:
    df[col+'_log'] = np.log1p(df[col])



    
# # #bi_variant

# plt.figure(figsize=(12,6))
# sns.boxplot(
#     data=df,
#     x='occupation',
#     y='annual_income'
# )
# plt.xticks(rotation=45)
# plt.show()

# plt.figure()
# sns.boxplot(
#     data=df,
#     x='education',
#     y='annual_income'
# )
# plt.show()


# plt.figure()
# sns.boxplot(
#     data=df,
#     x='customer_segment',
#     y='average_balance'
# )
# plt.show()


# plt.figure()
# sns.boxplot(
#     data=df,
#     x='risk_profile',
#     y='loan_amount'
# )
# plt.show()

# plt.figure(figsize=(8,5))
# sns.boxplot(
#     data=df,
#     x='marital_status',
#     y='expense_ratio'
# )
# plt.show()


#bivarint shows mostly normal


# #multi 


# plt.figure(figsize=(16,10))
# sns.heatmap(
#     df.select_dtypes(include='number').corr(),
#     annot=True,
#     cmap='coolwarm',
#     fmt=".2f"
# )

# plt.show()


# sns.pairplot(df[['annual_income_log','average_balance_log','credit_score','loan_amount_log']])
# plt.show()


#loan_amount has a distinct zero-inflated segment representing non-loan customers.

# plt.figure(figsize=(10,6))

# sns.scatterplot(
#     data=df,
#     x='annual_income',
#     y='average_balance',
#     hue='age',
#     alpha=0.6,
#     palette='viridis'
# )

# plt.title('Annual Income vs Average Balance (colored by Age)')
# plt.show()


product_cols = [
    'salary_account','savings_account','credit_card','personal_loan',
    'home_loan','car_loan','gold_loan','fixed_deposit',
    'recurring_deposit','mutual_fund','insurance','demat_account',
    'forex_card','travel_card'
]

df['product_count'] = df[product_cols].sum(axis=1)



# Droping duplicate column(high multicolinear +1)
df = df.drop(columns=['monthly_income'])

#flaging no-loan customers 
df['has_loan'] = (df['loan_amount'] > 0).astype(int)

# print(df.columns)

# Save cleaned dataset
df.to_csv('./datasets/ml_dataset_cleaned.csv', index=False)