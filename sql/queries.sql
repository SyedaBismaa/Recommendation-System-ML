SELECT COUNT(Customer_ID) AS total_ids
FROM customers

SELECT COUNT(Customer_ID) AS total_ids
FROM accounts

SELECT COUNT(Customer_ID) AS total_ids
FROM transactions



SELECT COUNT(*) AS missing_in_customers
FROM accounts a
WHERE a.customer_id NOT IN (SELECT c.customer_id FROM customers c)

SELECT COUNT(*) AS missing_in_customers
FROM Transactions t
WHERE t.customer_id NOT IN (SELECT c.customer_id FROM customers c)


SELECT COUNT(*) AS missing_in_customer
FROM Banking_Products bp
WHERE bp.customer_id NOT IN (SELECT c.customer_id FROM customers c)




    --transaction table converted

CREATE TABLE transaction_summary AS
SELECT
    customer_id,
    SUM(amount) AS total_amount,
    COUNT(transaction_id) AS total_transactions,
    AVG(amount) AS avg_amount,
    MAX(amount) AS max_amount,
    MIN(amount) AS min_amount
FROM transactions
GROUP BY customer_id;


--accounts and customers and transaction summary 

SELECT *
FROM customers c
INNER JOIN accounts a
    ON c.customer_id = a.customer_id
INNER JOIN banking_products bp
    ON c.customer_id = bp.customer_id
INNER JOIN transaction_summary ts
    ON c.customer_id = ts.customer_id;



CREATE TABLE customer_ml_dataset AS
SELECT

    -- Customer Details
    c.customer_id,
    c.age,
    c.gender,
    c.marital_status,
    c.education,
    c.occupation,
    c.state,
    c.employment_status,
    c.annual_income,
    c.monthly_income,
    c.customer_since,
    c.customer_segment,
    c.risk_profile,

    -- Account Details
    a.account_type,
    a.average_balance,
    a.monthly_expenses,
    a.monthly_savings,
    a.emi,
    a.loan_amount,
    a.credit_score,
    a.debt_to_income_ratio,
    a.number_of_accounts,
    a.digital_banking_usage,
    a.mobile_app_usage,
    a.atm_transactions,
    a.online_transactions,

    -- Banking Products
    bp.salary_account,
    bp.savings_account,
    bp.credit_card,
    bp.personal_loan,
    bp.home_loan,
    bp.car_loan,
    bp.gold_loan,
    bp.fixed_deposit,
    bp.recurring_deposit,
    bp.mutual_fund,
    bp.insurance,
    bp.demat_account,
    bp.forex_card,
    bp.travel_card,

    -- Engineered Transaction Features
    ts.total_amount,
    ts.total_transactions,
    ts.avg_amount,
    ts.max_amount,
    ts.min_amount

FROM customers c

INNER JOIN accounts a
ON c.customer_id = a.customer_id

INNER JOIN banking_products bp
ON c.customer_id = bp.customer_id

INNER JOIN transaction_summary ts
ON c.customer_id = ts.customer_id;



SELECT *
FROM customer_ml_dataset
LIMIT 10;