import pandas as pd
import sqlite3

# Load CSV
df = pd.read_csv("TurnerAI.csv")

# Strip spaces from all column names
df.columns = df.columns.str.strip()

# Fix date format in From_Date column (to YYYY-MM-DD)
if "From_Date" in df.columns:
    df["From_Date"] = pd.to_datetime(df["From_Date"], errors='coerce').dt.strftime('%Y-%m-%d')

# Save to SQLite
conn = sqlite3.connect("sales_data.db")
df.to_sql("sales", conn, if_exists="replace", index=False)
conn.close()

print("✅ Cleaned data and imported into SQLite successfully.")