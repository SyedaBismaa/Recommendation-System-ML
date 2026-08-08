

-- DROP TABLE IF EXISTS banking_products CASCADE;
-- DROP TABLE IF EXISTS transactions CASCADE;
-- DROP TABLE IF EXISTS accounts CASCADE;
-- DROP TABLE IF EXISTS customers CASCADE;



CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    age INT,
    gender VARCHAR(20),
    marital_status VARCHAR(50),
    education VARCHAR(50),
    occupation VARCHAR(50),
    city VARCHAR(50),
    state VARCHAR(50),
    employment_status VARCHAR(50),
    annual_income DECIMAL(10,2),
    monthly_income DECIMAL(10,2),
    customer_since INT,
    customer_segment VARCHAR(50),
    risk_profile VARCHAR(50)
);




CREATE TABLE accounts (
    account_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20),
    account_type VARCHAR(50),
    average_balance DECIMAL(18,2),
    monthly_expenses DECIMAL(18,2),
    monthly_savings DECIMAL(18,2),
    emi DECIMAL(18,2),
    loan_amount DECIMAL(18,2),
    credit_score INT,
    debt_to_income_ratio DECIMAL(10,2),
    number_of_accounts INT,
    digital_banking_usage INT,
    mobile_app_usage INT,
    atm_transactions INT,
    online_transactions INT,
    FOREIGN KEY (customer_id)
    REFERENCES customers(customer_id)
);




CREATE TABLE transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20),
    transaction_date DATE,
    transaction_type VARCHAR(50),
    merchant_category VARCHAR(100),
    transaction_channel VARCHAR(50),
    amount DECIMAL(10,2),
    FOREIGN KEY (customer_id)
    REFERENCES customers(customer_id)
);



CREATE TABLE banking_products (
    customer_id VARCHAR(20) PRIMARY KEY,
    salary_account BOOLEAN,
    savings_account BOOLEAN,
    credit_card BOOLEAN,
    personal_loan BOOLEAN,
    home_loan BOOLEAN,
    car_loan BOOLEAN,
    gold_loan BOOLEAN,
    fixed_deposit BOOLEAN,
    recurring_deposit BOOLEAN,
    mutual_fund BOOLEAN,
    insurance BOOLEAN,
    demat_account BOOLEAN,
    forex_card BOOLEAN,
    travel_card BOOLEAN,
    FOREIGN KEY (customer_id)
    REFERENCES customers(customer_id)
);