# Banking Product Recommendation System

An end-to-end machine learning system that segments bank customers based on behavioral and financial data, then recommends relevant banking products using association rule mining.

The project focuses on reducing demographic bias by relying primarily on behavioral and financial signals for customer segmentation and recommendations.

**Live App:** [https://recommendation-system-ml-tldmkmp6qcegg65onftsro.streamlit.app/]

---

## Overview

The system takes customer financial and behavioral data, groups similar customers into segments using clustering, mines product ownership patterns within each segment using association rule mining, and recommends products a customer is likely to need but does not already have.

---

## Project Pipeline

### 1. Data Cleaning

- Missing value treatment using median imputation for numerical columns and mode imputation for categorical columns.
- Outlier treatment using percentile capping (1st–99th percentile) on heavily right-skewed financial features such as income, balance, loan amount, and expenses.
- Log transformation using `log1p` after capping to reduce skewness and improve the suitability of financial features for modeling.

### 2. Exploratory Data Analysis

- **Univariate analysis:** Distribution plots and boxplots for numerical features.
- **Bivariate analysis:** Relationship analysis between categorical segments such as occupation and risk profile and financial variables.
- **Multivariate analysis:** Correlation heatmap, pairplots, and scatterplots to identify relationships and multicollinearity.
- Identified and removed the fully redundant `monthly_income` feature due to its perfect correlation with `annual_income`.

### 3. Feature Engineering

Created features based on customer financial and behavioral information:

- `savings_ratio`
- `expense_ratio`
- `loan_burden`
- `balance_to_income`
- `product_count`
- `loan_count`
- `investment_count`
- `digital_engagement`
- `has_loan`

Additional preprocessing included:

- One-hot encoding for nominal categorical variables.
- Custom binning for age and customer tenure.

### 4. Bias Mitigation

Demographic attributes such as:

- Age
- Gender
- Marital Status
- Education
- Occupation

were deliberately excluded from the clustering and recommendation feature set.

The recommendation system therefore relies primarily on financial and behavioral signals rather than demographic characteristics.

Demographic fields are retained separately and can be used for potential post-hoc fairness analysis.

### 5. Dimensionality Reduction

- Standardized the feature set using `StandardScaler`.
- Applied **Principal Component Analysis (PCA)**.
- Retained approximately **90% of the variance**.
- Reduced approximately 30 behavioral features to **17 principal components** before clustering.

### 6. Customer Segmentation

**K-Means clustering** was used as the primary customer segmentation algorithm.

- The optimal cluster count was selected as **k = 5** using the Elbow Method.
- Silhouette Score was used as an additional validation metric.
- Silhouette scores were approximately **0.15–0.17**, indicating that customer segments have some overlap.

**DBSCAN** was also evaluated as an alternative clustering algorithm. It did not produce meaningful density-based clusters on this dataset, with more than 96% of observations falling into a single cluster. Therefore, K-Means was selected as the primary segmentation approach.

#### Clustering Visualizations

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/6eb645c5-6970-490a-90e4-76bde014d6b3" />

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/b63799b5-59cd-45fe-9bc9-3aaddde9d22c" />


### 7. Association Rule Mining

**FP-Growth** was applied within each customer segment to identify frequently co-owned banking products.

The generated association rules were evaluated using:

- **Support** — How frequently an itemset appears in the dataset.
- **Confidence** — How often the consequent occurs when the antecedent occurs.
- **Lift** — How strongly two products are associated compared with random occurrence.

FP-Growth was selected over Apriori because it provides better computational efficiency for the itemset size used in this project.

### 8. Recommendation Engine

The recommendation engine uses a hybrid approach.

#### Primary Recommendation

FP-Growth association rules are used to recommend products based on:

- The customer's existing products.
- The customer's assigned customer segment.
- Rule confidence and lift.

#### Fallback Recommendation

If no suitable association rule is found, the system recommends popular products within the customer's segment.

This provides a fallback mechanism for customers where no direct association rule is available.

### 9. Deployment

The trained components were serialized using `joblib`, including:

- StandardScaler
- PCA model
- K-Means model
- Association rules

The application was deployed using **Streamlit** and supports:

- Single-customer recommendations.
- Customer segment assignment.
- Product recommendations.

---

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Clustering | K-Means, DBSCAN |
| Dimensionality Reduction | PCA |
| Association Rule Mining | mlxtend, FP-Growth |
| Deployment | Streamlit |
| Model Persistence | Joblib |

---

## Project Structure

```text
Recommendation_System/
│
├── datasets/
│   ├── ml_dataset.csv
│   ├── model_ready_ds.csv
│   └── final_clustered_ds.csv
│
├── pkl/
│   ├── scaler.pkl
│   ├── pca_model.pkl
│   ├── kmeans_model.pkl
│   └── cluster_rules.pkl
│
├── plots.py
├── cleaning.py
├── models.py
├── app.py
├── requirements.txt
└── README.md
```

### File Description

- `plots.py` — Generates EDA visualizations.
- `cleaning.py` — Handles data cleaning, feature engineering, and preparation of the model-ready dataset.
- `models.py` — Performs PCA, clustering, FP-Growth, and recommendation logic.
- `app.py` — Streamlit application.
- `pkl/` — Stores serialized models and recommendation rules.
- `datasets/` — Contains processed datasets used throughout the project.

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Recommendation_System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Project Goal

The goal of this project is to demonstrate an end-to-end approach to banking customer segmentation and product recommendation using unsupervised machine learning and association rule mining.

The project combines data preprocessing, feature engineering, dimensionality reduction, clustering, association rule mining, recommendation logic, and deployment into a single machine learning workflow.
