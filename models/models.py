import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#DBscan 
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

#Fp growth
from mlxtend.frequent_patterns import fpgrowth,association_rules

#joblib 
import joblib

df = pd.read_csv('./datasets/model_ready_ds.csv')

# SET A
cluster_features = [
    "customer_since","monthly_savings","emi","credit_score","debt_to_income_ratio",
    "credit_card","fixed_deposit","recurring_deposit","mutual_fund","insurance",
    "demat_account","forex_card","travel_card","total_transactions","avg_amount",
    "annual_income_log","average_balance_log","loan_amount_log","monthly_expenses_log",
    "has_loan","risk_profile_High","risk_profile_Low","risk_profile_Medium",
    "savings_ratio","expense_ratio","loan_burden","balance_to_income",
    "product_count","loan_count","investment_count","digital_engagement"
]

X = df[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# plt.plot(range(2, 11), wcss, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.title("Elbow Method")
# plt.show()


for i in range(2,8):
    kmeans=KMeans(n_clusters=i, random_state=42, n_init=10)
    labels=kmeans.fit_predict(X_scaled)
    score=silhouette_score(X_scaled,labels)
    # print(f"k={i}, silhouette={score:.3f}")   #sillhout score is very less (overlapping clusters )

pca=PCA(n_components=0.90)
X_pca=pca.fit_transform(X_scaled)
print(X_pca.shape)

for i in range(2,8):
    kmeans=KMeans(n_clusters=i, random_state=42, n_init=10)
    labels=kmeans.fit_predict(X_scaled)
    score=silhouette_score(X_pca,labels)
    # print(f"k={i}, silhouette={score:.3f}") 

kmeans_final = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_pca)
# print(df['cluster'].value_counts())
df.groupby('cluster')[cluster_features].mean()



#FP growth

product_cols=[
    'salary_account','savings_account','credit_card','personal_loan',
    'home_loan','car_loan','gold_loan','fixed_deposit',
    'recurring_deposit','mutual_fund','insurance','demat_account',
    'forex_card','travel_card'
]

all_rules={}

for c in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster']==c][product_cols]
    cluster_data = cluster_data.astype(bool)

    frequent_itemsets = fpgrowth(cluster_data,min_support=0.05,use_colnames=True)

    if frequent_itemsets.empty:
        print(f"Cluster {c}: no frequent itemsets found, try lowering min_support")
        continue

    rules= association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules=rules.sort_values('lift', ascending=False)


    all_rules[c]=rules
    # print(f"\n--- Cluster {c} ({len(cluster_data)} customers) ---")
    # print(rules[['antecedents','consequents','support','confidence','lift']].head(5))


df[product_cols].mean().sort_values(ascending=False)
df.groupby('cluster')[product_cols].mean().T



def recommend_products(customer_products, cluster_id, all_rules, df, product_cols, top_n=3):
    customer_products = set(customer_products)
    recommendations = []

    if cluster_id in all_rules:
        rules = all_rules[cluster_id].sort_values(['confidence','lift'], ascending=False)
        for _, row in rules.iterrows():
            antecedents = set(row['antecedents'])
            consequents = set(row['consequents'])
            if antecedents.issubset(customer_products) and not consequents.issubset(customer_products):
                for item in consequents:
                    if item not in customer_products and item not in recommendations:
                        recommendations.append(item)
            if len(recommendations) >= top_n:
                break

    # fallback: 
    if len(recommendations) < top_n:
        popular = df[df['cluster']==cluster_id][product_cols].mean().sort_values(ascending=False)
        for item in popular.index:
            if item not in customer_products and item not in recommendations:
                recommendations.append(item)
            if len(recommendations) >= top_n:
                break

    return recommendations[:top_n]


joblib.dump(kmeans_final, 'kmeans_model.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(pca,'pca_model.pkl')
joblib.dump(all_rules,'cluster_rules.pkl')
df.to_csv('./datasets/final_clustered_ds.csv',index=False)




#testing--
# sample_customer = {'salary_account':1,'credit_card':1,'travel_card':0,'forex_card':0}  # example
# customer_products = [k for k,v in sample_customer.items() if v==1]

# recs = recommend_products(customer_products, cluster_id=1, all_rules=all_rules, df=df, product_cols=product_cols)
# print(recs)







#DBScan

# neighbors=NearestNeighbors(n_neighbors=5)
# neighbors_fit=neighbors.fit(X_pca)
# distances ,indices = neighbors_fit.kneighbors(X_pca)
# distances=np.sort(distances[:,4])

# plt.plot(distances)
# plt.ylabel("5NN distance")
# plt.title("K_distance graph")
# # plt.show()   # acc to this best ep will be 3.5

# dbscan = DBSCAN(eps=3.5, min_samples=5)
# df['dbscan_cluster'] = dbscan.fit_predict(X_pca)
# print(df['dbscan_cluster'].value_counts())

# # OP   , droping dp scan 
# #  0    4834
# # -1     164