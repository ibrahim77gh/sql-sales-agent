from langchain_community.utilities import SQLDatabase
import os
import hashlib
from sqlalchemy import text

SYSTEM_PROMPT = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer in a clear, user-friendly format.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    database.

    To start you should ALWAYS look at the tables in the database to see what you
    can query. Do NOT skip this step.

    Then you should query the schema of the most relevant tables.

    When presenting query results, summarize or format them nicely in your final response markdown.
    Do not just output the raw query result string.
    """

def get_database_schema_hash():
    """Generate a hash of the current database schema to detect changes (MSSQL version)."""
    try:
        db_uri = os.getenv('DB_URI')
        if not db_uri:
            return None, "DB_URI environment variable not set. Cannot connect to database."
        
        db = SQLDatabase.from_uri(db_uri)
        engine = db._engine
        
        # Get table and column info using MSSQL-compatible views
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS
                ORDER BY TABLE_NAME, ORDINAL_POSITION
            """))

            schema_info = result.fetchall()
        
        # Format schema into a consistent string
        schema_string = "\n".join([
            f"{row[0]}|{row[1]}|{row[2]}|{row[3]}|{row[4]}"
            for row in schema_info
        ])

        # Create and return hash
        return hashlib.md5(schema_string.encode()).hexdigest(), None

    except Exception as e:
        print(f"Error getting schema hash: {e}")
        return None, str(e)
    

def get_sample_training_data():
    """Return sample training data for the model with MSSQL-specific syntax."""
    return [
        {
            "question": "What are the top 5 regions by total sales?",
            "sql": "SELECT TOP 5 [region_name], SUM([Total Sales]) as total_sales FROM ConsolidateData_PBI GROUP BY [region_name] ORDER BY total_sales DESC"
        },
        {
            "question": "What are the top 5 companies by protection sales?",
            "sql": "SELECT TOP 5 [Company_Name], SUM([Protection SALES]) as total_protection_sales FROM ConsolidateData_PBI GROUP BY [Company_Name] ORDER BY total_protection_sales DESC"
        },
        {
            "question": "Show sales trends by date and region",
            "sql": "SELECT [From_Date], [region_name], SUM([Total Sales]) as total_sales FROM ConsolidateData_PBI GROUP BY [From_Date], [region_name] ORDER BY [From_Date]"
        },
        {
            "question": "Show sales data for June 2025",
            "sql": "SELECT * FROM ConsolidateData_PBI WHERE YEAR([From_Date]) = 2025 AND MONTH([From_Date]) = 6"
        },
        {
            "question": "What is the total sales for June 2025?",
            "sql": "SELECT SUM([Total Sales]) as total_sales FROM ConsolidateData_PBI WHERE YEAR([From_Date]) = 2025 AND MONTH([From_Date]) = 6"
        },
        {
            "question": "Get sales data for a specific month and year",
            "sql": "SELECT * FROM ConsolidateData_PBI WHERE YEAR([From_Date]) = 2025 AND MONTH([From_Date]) = 12"
        },
        {
            "question": "Show monthly sales totals for 2024",
            "sql": "SELECT YEAR([From_Date]) as sales_year, MONTH([From_Date]) as sales_month, SUM([Total Sales]) as monthly_total FROM ConsolidateData_PBI WHERE YEAR([From_Date]) = 2024 GROUP BY YEAR([From_Date]), MONTH([From_Date]) ORDER BY sales_year, sales_month"
        },
        {
            "question": "Show monthly sales totals for 2025",
            "sql": "SELECT YEAR([From_Date]) as sales_year, MONTH([From_Date]) as sales_month, SUM([Total Sales]) as monthly_total FROM ConsolidateData_PBI WHERE YEAR([From_Date]) = 2025 GROUP BY YEAR([From_Date]), MONTH([From_Date]) ORDER BY sales_year, sales_month"
        },
        {
            "question": "Get sales data for last 30 days",
            "sql": "SELECT * FROM ConsolidateData_PBI WHERE [From_Date] >= DATEADD(day, -30, GETDATE())"
        },
        {
            "question": "What are the sales for the current month?",
            "sql": "SELECT SUM([Total Sales]) as current_month_sales FROM ConsolidateData_PBI WHERE YEAR([From_Date]) = YEAR(GETDATE()) AND MONTH([From_Date]) = MONTH(GETDATE())"
        },
        {
            "question": "Show sales by month and year",
            "sql": "SELECT YEAR([From_Date]) as year, MONTH([From_Date]) as month, SUM([Total Sales]) as monthly_sales FROM ConsolidateData_PBI GROUP BY YEAR([From_Date]), MONTH([From_Date]) ORDER BY year, month"
        },
        {
            "question": "Get sales data between two dates",
            "sql": "SELECT * FROM ConsolidateData_PBI WHERE [From_Date] BETWEEN '2025-01-01' AND '2025-06-30'"
        },
        {
            "question": "What is the average sales per day?",
            "sql": "SELECT AVG([Total Sales]) as avg_daily_sales FROM ConsolidateData_PBI"
        },
        {
            "question": "Show total sales by quarter",
            "sql": "SELECT YEAR([From_Date]) as year, DATEPART(quarter, [From_Date]) as quarter, SUM([Total Sales]) as quarterly_sales FROM ConsolidateData_PBI GROUP BY YEAR([From_Date]), DATEPART(quarter, [From_Date]) ORDER BY year, quarter"
        }
    ]
