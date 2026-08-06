import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

query = "SELECT * FROM customer_ml_dataset"

df = pd.read_sql(query, engine)

df.to_csv("datasets/customer_ml_dataset.csv", index=False)

print("✅ Dataset exported successfully!")
print(df.shape)