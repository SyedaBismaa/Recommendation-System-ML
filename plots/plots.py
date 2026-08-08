import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./datasets/ml_dataset.csv")

num_cols = ['annual_income','monthly_income','average_balance','loan_amount',
            'credit_score','total_amount','monthly_expenses']

for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()

for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# bivariate
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='occupation', y='annual_income')
plt.xticks(rotation=45)
plt.show()

plt.figure()
sns.boxplot(data=df, x='education', y='annual_income')
plt.show()

plt.figure()
sns.boxplot(data=df, x='customer_segment', y='average_balance')
plt.show()

plt.figure()
sns.boxplot(data=df, x='risk_profile', y='loan_amount')
plt.show()

# multivariate
plt.figure(figsize=(16,10))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

sns.pairplot(df[['annual_income','average_balance','credit_score','loan_amount']])
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='annual_income', y='average_balance', hue='age', alpha=0.6, palette='viridis')
plt.title('Annual Income vs Average Balance (colored by Age)')
plt.show()