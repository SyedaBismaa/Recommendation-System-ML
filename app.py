import pandas as pd
pd.set_option('future.infer_string', False)
import numpy as np
import streamlit as st
import joblib


scaler = joblib.load('./pkl/scaler.pkl')
pca = joblib.load('./pkl/pca_model.pkl')
kmeans=joblib.load('./pkl/kmeans_model.pkl')
all_rules=joblib.load('./pkl/cluster_rules.pkl')

df=pd.read_csv('./datasets/final_clustered_ds.csv')

print(df.head(5))



product_cols = [
    'salary_account','savings_account','credit_card','personal_loan',
    'home_loan','car_loan','gold_loan','fixed_deposit',
    'recurring_deposit','mutual_fund','insurance','demat_account',
    'forex_card','travel_card'
]

cluster_features = [
    "customer_since","monthly_savings","emi","credit_score","debt_to_income_ratio",
    "credit_card","fixed_deposit","recurring_deposit","mutual_fund","insurance",
    "demat_account","forex_card","travel_card","total_transactions","avg_amount",
    "annual_income_log","average_balance_log","loan_amount_log","monthly_expenses_log",
    "has_loan","risk_profile_High","risk_profile_Low","risk_profile_Medium",
    "savings_ratio","expense_ratio","loan_burden","balance_to_income",
    "product_count","loan_count","investment_count","digital_engagement"
]

st.title("🏦 Banking Product Recommendation System")
st.caption("Enter customer details to get personalized product recommendations")

with st.form("customer_form"):

    st.markdown("### 💰 Financial Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        monthly_income = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0)
        monthly_savings = st.number_input("Monthly Savings", min_value=0.0, value=5000.0, step=500.0)
    with c2:
        monthly_expenses = st.number_input("Monthly Expenses", min_value=0.0, value=20000.0, step=500.0)
        average_balance = st.number_input("Average Balance", min_value=0.0, value=100000.0, step=1000.0)
    with c3:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        avg_amount = st.number_input("Avg Transaction Amount", min_value=0.0, value=5000.0, step=500.0)

    st.markdown("### 🏠 Loan Details")
    c4, c5, c6 = st.columns(3)
    with c4:
        has_loan_input = st.checkbox("Does customer have a loan?")
    loan_amount = 0.0
    emi = 0.0
    if has_loan_input:
        with c5:
            loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0, step=1000.0)
        with c6:
            emi = st.number_input("Monthly EMI", min_value=0.0, value=5000.0, step=500.0)

    st.markdown("### 📊 Account & Risk Info")
    c7, c8, c9 = st.columns(3)
    with c7:
        customer_since = st.number_input("Customer Since (years)", min_value=0, value=3)
    with c8:
        risk_profile = st.selectbox("Risk Profile", ["Low","Medium","High"])
    with c9:
        digital_engagement = st.slider("Digital Engagement", 0.0, 1.0, 0.5)

    debt_to_income_ratio = round((emi * 12) / monthly_income, 2) if monthly_income else 0.0
    total_transactions = st.slider("Monthly Transactions (approx)", 0, 100, 20)

    st.markdown("### 💳 Existing Products")
    product_inputs = {}
    p_cols = st.columns(4)
    for i, p in enumerate(product_cols):
        product_inputs[p] = p_cols[i % 4].checkbox(p.replace('_',' ').title())

    submitted = st.form_submit_button("Get Recommendations", use_container_width=True)

# ---------------- Process on submit ----------------
if submitted:

    annual_income = monthly_income * 12
    has_loan = 1 if loan_amount > 0 else 0

    annual_income_log = np.log1p(annual_income)
    average_balance_log = np.log1p(average_balance)
    loan_amount_log = np.log1p(loan_amount)
    monthly_expenses_log = np.log1p(monthly_expenses)

    savings_ratio = monthly_savings / monthly_income if monthly_income else 0
    expense_ratio = monthly_expenses / monthly_income if monthly_income else 0
    loan_burden = loan_amount / annual_income if annual_income else 0
    balance_to_income = average_balance / annual_income if annual_income else 0

    product_count = sum(product_inputs.values())
    loan_cols = ['personal_loan','home_loan','car_loan','gold_loan']
    investment_cols = ['fixed_deposit','recurring_deposit','mutual_fund','insurance','demat_account']
    loan_count = sum(product_inputs[c] for c in loan_cols)
    investment_count = sum(product_inputs[c] for c in investment_cols)

    risk_High = 1 if risk_profile == "High" else 0
    risk_Low = 1 if risk_profile == "Low" else 0
    risk_Medium = 1 if risk_profile == "Medium" else 0

    row = {
        "customer_since": customer_since,
        "monthly_savings": monthly_savings,
        "emi": emi,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_card": int(product_inputs['credit_card']),
        "fixed_deposit": int(product_inputs['fixed_deposit']),
        "recurring_deposit": int(product_inputs['recurring_deposit']),
        "mutual_fund": int(product_inputs['mutual_fund']),
        "insurance": int(product_inputs['insurance']),
        "demat_account": int(product_inputs['demat_account']),
        "forex_card": int(product_inputs['forex_card']),
        "travel_card": int(product_inputs['travel_card']),
        "total_transactions": total_transactions,
        "avg_amount": avg_amount,
        "annual_income_log": annual_income_log,
        "average_balance_log": average_balance_log,
        "loan_amount_log": loan_amount_log,
        "monthly_expenses_log": monthly_expenses_log,
        "has_loan": has_loan,
        "risk_profile_High": risk_High,
        "risk_profile_Low": risk_Low,
        "risk_profile_Medium": risk_Medium,
        "savings_ratio": savings_ratio,
        "expense_ratio": expense_ratio,
        "loan_burden": loan_burden,
        "balance_to_income": balance_to_income,
        "product_count": product_count,
        "loan_count": loan_count,
        "investment_count": investment_count,
        "digital_engagement": digital_engagement,
    }

    input_df = pd.DataFrame([row])[cluster_features]

    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    cluster_id = kmeans.predict(input_pca)[0]

    st.success(f"Customer assigned to Cluster {cluster_id}")

    customer_products = [p for p, v in product_inputs.items() if v]

    def recommend_products(customer_products, cluster_id, all_rules, df, product_cols, top_n=3):
        customer_products = set(customer_products)
        recommendations = []
        if cluster_id in all_rules:
            rules = all_rules[cluster_id].sort_values(['confidence','lift'], ascending=False)
            for _, r in rules.iterrows():
                antecedents = set(r['antecedents'])
                consequents = set(r['consequents'])
                if antecedents.issubset(customer_products) and not consequents.issubset(customer_products):
                    for item in consequents:
                        if item not in customer_products and item not in recommendations:
                            recommendations.append(item)
                if len(recommendations) >= top_n:
                    break
        if len(recommendations) < top_n:
            popular = df[df['cluster']==cluster_id][product_cols].mean().sort_values(ascending=False)
            for item in popular.index:
                if item not in customer_products and item not in recommendations:
                    recommendations.append(item)
                if len(recommendations) >= top_n:
                    break
        return recommendations[:top_n]

    recs = recommend_products(customer_products, cluster_id, all_rules, df, product_cols)

    st.markdown("### 🎯 Recommended Products")
    rec_cols = st.columns(len(recs)) if recs else [st]
    for col, r in zip(rec_cols, recs):
        col.info(f"**{r.replace('_',' ').title()}**")
