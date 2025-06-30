import sqlite3
import pandas as pd

# 1. Connect to the database file
conn = sqlite3.connect("sales_data.db")
cursor = conn.cursor()



#conn = sqlite3.connect("chroma.sqlite3")


#cursor.execute(" SELECT [region_name], SUM([Gross Margin]) as total_gross_margin FROM sales WHERE strftime('%m-%d', [From_Date]) = '05-05' GROUP BY [region_name]")


# 2. View sample metadata
#meta = pd.read_sql("SELECT SUM([Avg Ticket Sale]) AS total_avg_ticket_sale FROM sales WHERE strftime('%Y-%m', [From_Date]) = '2025-05'", conn)
#print("Metadata sample:\n", meta)

# Load entire table or specific query into a DataFrame
query = """
SELECT SUM([Traffic Count]) AS total_traffic_count FROM sales 
WHERE strftime('%Y-%m', [From_Date]) = '2025-05'
"""

df = pd.read_sql(query, conn)
print(df)

# conn.close()
# df = pd.read_csv("TurnerAI.csv")
# print("Before:", df.columns.tolist())

# df.columns = df.columns.str.strip()
# print("After:", df.columns.tolist())

# # 3. Rebuild SQLite table cleanly
# conn = sqlite3.connect("sales_data.db")
# df.to_sql("sales", conn, if_exists="replace", index=False)
# conn.close()