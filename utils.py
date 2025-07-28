from langchain_community.utilities import SQLDatabase
import os
import hashlib
from sqlalchemy import text

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

SCHEMA_HASH_FILE = "schema_hash.txt"  # File to track schema changes

def has_schema_changed():
    """Check if the database schema has changed since last training."""
    current_hash, error = get_database_schema_hash()  # Unpack here too
    
    if error or current_hash is None:
        return True  # Assume changed if we can't get hash
    
    try:
        if os.path.exists(SCHEMA_HASH_FILE):
            with open(SCHEMA_HASH_FILE, 'r') as f:
                stored_hash = f.read().strip()
            return current_hash != stored_hash
        else:
            return True
    except Exception:
        return True


def save_schema_hash():
    """Save the current schema hash."""
    current_hash, error = get_database_schema_hash()  # Unpack the tuple
    
    if error:
        print(f"Error getting schema hash: {error}")
        return
        
    if current_hash:
        try:
            with open(SCHEMA_HASH_FILE, 'w') as f:
                f.write(current_hash)  # Now writing just the hash string
        except Exception as e:
            print(f"Error saving schema hash: {e}")


def train_vanna_dynamically(vn, add_sample_data: bool = True):
    """Dynamically trains Vanna agent using schema introspection + optional sample data."""
    # 1. Fetch info schema
    df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

    # 2. Generate training plan and train
    plan = vn.get_training_plan_generic(df_information_schema)
    vn.train(plan=plan)

    # 3. Add documentation
    vn.train(documentation="""
    This is a Microsoft SQL Server (MSSQL) sales database containing customer transaction data.

    CRITICAL SYNTAX REQUIREMENTS:
    - This is Microsoft SQL Server, NOT SQLite or any other database
    - NEVER use SQLite functions like strftime(), datetime(), etc.
    - Use MSSQL syntax exclusively

    DATE FUNCTIONS (MSSQL ONLY):
    - Use YEAR([column]) to extract year
    - Use MONTH([column]) to extract month  
    - Use DAY([column]) to extract day
    - Use DATEPART(quarter, [column]) for quarters
    - Use GETDATE() for current date/time
    - Use DATEADD(interval, number, date) for date arithmetic
    - Use BETWEEN for date ranges

    QUERY SYNTAX:
    - Use TOP N instead of LIMIT N
    - Use square brackets [column_name] for column names with spaces
    - Use proper MSSQL aggregate functions
    """)

    # # 4. Optional: Train with sample Q&A pairs
    # if add_sample_data:
    #     df_tables = vn.run_sql("""
    #         SELECT TABLE_NAME
    #         FROM INFORMATION_SCHEMA.TABLES
    #         WHERE TABLE_TYPE = 'BASE TABLE'
    #     """)
    #     if not df_tables.empty:
    #         target_table = df_tables['TABLE_NAME'].iloc[0]
    #         sample_data = get_sample_training_data(target_table)
    #         for item in sample_data:
    #             vn.train(question=item["question"], sql=item["sql"])

    # 5. Save hash
    save_schema_hash()
