import os
import pyodbc
from sqlalchemy import create_engine, text

def diagnose_connection():
    """Diagnose database connection issues."""
    print("=== Database Connection Diagnostics ===\n")
    
    # Check environment variable
    db_uri = os.getenv('DB_URI')
    print(f"1. DB_URI from environment: {db_uri}")
    
    if not db_uri:
        print("❌ DB_URI environment variable not found!")
        return
    
    # Check for quotes in the URI
    if db_uri.startswith("'") and db_uri.endswith("'"):
        print("⚠️  Warning: DB_URI has quotes around it. Remove quotes from .env file")
        db_uri = db_uri.strip("'")
        print(f"   Cleaned URI: {db_uri}")
    
    # Check available ODBC drivers
    print("\n2. Available ODBC Drivers:")
    try:
        drivers = pyodbc.drivers()
        sql_server_drivers = [d for d in drivers if 'SQL Server' in d]
        print(f"   Found {len(sql_server_drivers)} SQL Server drivers:")
        for driver in sql_server_drivers:
            print(f"   - {driver}")
        
        if not sql_server_drivers:
            print("   ❌ No SQL Server ODBC drivers found!")
            return
            
    except Exception as e:
        print(f"   ❌ Error checking drivers: {e}")
        return
    
    # Test direct pyodbc connection
    print("\n3. Testing direct pyodbc connection:")
    try:
        # Extract connection parts for pyodbc
        conn_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-14S8BC7;DATABASE=TurnerAI;Trusted_Connection=yes;"
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        conn.close()
        print("   ✅ Direct pyodbc connection successful!")
    except Exception as e:
        print(f"   ❌ Direct pyodbc connection failed: {e}")
        
        # Try alternative driver names
        alt_drivers = [
            "ODBC Driver 18 for SQL Server",
            "ODBC Driver 13 for SQL Server",
            "SQL Server Native Client 11.0",
            "SQL Server"
        ]
        
        print("   Trying alternative drivers...")
        for alt_driver in alt_drivers:
            if alt_driver in drivers:
                try:
                    alt_conn_str = f"DRIVER={{{alt_driver}}};SERVER=DESKTOP-14S8BC7;DATABASE=TurnerAI;Trusted_Connection=yes;"
                    conn = pyodbc.connect(alt_conn_str)
                    conn.close()
                    print(f"   ✅ Alternative driver '{alt_driver}' works!")
                    print(f"   Suggested DB_URI: mssql+pyodbc://@DESKTOP-14S8BC7/TurnerAI?driver={alt_driver.replace(' ', '+')}&trusted_connection=yes")
                    break
                except Exception as alt_e:
                    print(f"   ❌ '{alt_driver}' failed: {alt_e}")
    
    # Test SQLAlchemy connection
    print("\n4. Testing SQLAlchemy connection:")
    try:
        engine = create_engine(db_uri)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            result.fetchone()
        print("   ✅ SQLAlchemy connection successful!")
    except Exception as e:
        print(f"   ❌ SQLAlchemy connection failed: {e}")
        
        # Suggest URL encoding fix
        if "Data source name not found" in str(e):
            print("   💡 Try URL encoding the driver name:")
            encoded_uri = db_uri.replace("ODBC Driver 17 for SQL Server", "ODBC+Driver+17+for+SQL+Server")
            print(f"   Suggested: {encoded_uri}")
    
    # Test database access
    print("\n5. Testing database table access:")
    try:
        engine = create_engine(db_uri)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
            """))
            tables = result.fetchall()
            print(f"   ✅ Found {len(tables)} tables in database")
            if tables:
                print("   Tables:")
                for table in tables[:5]:  # Show first 5 tables
                    print(f"   - {table[0]}")
                if len(tables) > 5:
                    print(f"   ... and {len(tables) - 5} more")
    except Exception as e:
        print(f"   ❌ Database table access failed: {e}")

# Alternative connection strings to try
def get_alternative_connection_strings():
    """Generate alternative connection strings to try."""
    base_server = "DESKTOP-14S8BC7"
    database = "TurnerAI"
    
    alternatives = [
        # URL encoded version
        f"mssql+pyodbc://@{base_server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes",
        
        # With different driver versions
        f"mssql+pyodbc://@{base_server}/{database}?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes",
        f"mssql+pyodbc://@{base_server}/{database}?driver=SQL+Server+Native+Client+11.0&trusted_connection=yes",
        f"mssql+pyodbc://@{base_server}/{database}?driver=SQL+Server&trusted_connection=yes",
        
        # With additional parameters
        f"mssql+pyodbc://@{base_server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes&autocommit=true",
    ]
    
    print("\n=== Alternative Connection Strings to Try ===")
    for i, alt in enumerate(alternatives, 1):
        print(f"{i}. {alt}")

if __name__ == "__main__":
    diagnose_connection()
    get_alternative_connection_strings()