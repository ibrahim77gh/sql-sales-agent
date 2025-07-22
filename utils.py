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
    

def get_sample_training_data(table_name, sales_column="Total Sales", date_column="From_Date", region_column="region_name", company_column="Company_Name", protection_column="Protection SALES"):
    """Generate dynamic training data based on provided table and column names."""
    return [
        {
            "question": "What are the top 5 regions by total sales?",
            "sql": f"SELECT TOP 5 [{region_column}], SUM([{sales_column}]) as total_sales FROM {table_name} GROUP BY [{region_column}] ORDER BY total_sales DESC"
        },
        {
            "question": "What are the top 5 companies by protection sales?",
            "sql": f"SELECT TOP 5 [{company_column}], SUM([{protection_column}]) as total_protection_sales FROM {table_name} GROUP BY [{company_column}] ORDER BY total_protection_sales DESC"
        },
        {
            "question": "Show sales trends by date and region",
            "sql": f"SELECT [{date_column}], [{region_column}], SUM([{sales_column}]) as total_sales FROM {table_name} GROUP BY [{date_column}], [{region_column}] ORDER BY [{date_column}]"
        },
        {
            "question": "Show sales data for June 2025",
            "sql": f"SELECT * FROM {table_name} WHERE YEAR([{date_column}]) = 2025 AND MONTH([{date_column}]) = 6"
        },
        {
            "question": "Get sales data for last 30 days",
            "sql": f"SELECT * FROM {table_name} WHERE [{date_column}] >= DATEADD(day, -30, GETDATE())"
        },
        {
            "question": "What are the sales for the current month?",
            "sql": f"SELECT SUM([{sales_column}]) as current_month_sales FROM {table_name} WHERE YEAR([{date_column}]) = YEAR(GETDATE()) AND MONTH([{date_column}]) = MONTH(GETDATE())"
        }
    ]
