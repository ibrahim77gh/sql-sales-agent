import time
import streamlit as st
import os
from dotenv import load_dotenv
import sqlite3
import pandas as pd
from streamlit_cookies_controller import CookieController
from dateutil.parser import parse as parse_datetime_string
import json
import datetime
import plotly.express as px
import plotly.graph_objects as go
go.Figure.show = lambda *args, **kwargs: None
import hashlib
import re

# Vanna AI imports
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

load_dotenv()

COOKIE_NAME = "login_state"
SCHEMA_HASH_FILE = "schema_hash.txt"  # File to track schema changes

# Sensitive keywords to block
SENSITIVE_KEYWORDS = [
    'schema', 'table', 'column', 'database', 'structure', 'ddl', 'create table',
    'alter table', 'drop table', 'sqlite_master', 'pragma', 'table_info',
    'describe', 'show tables', 'show columns', 'information_schema',
    'sys.', 'metadata', 'system', 'admin', 'configuration', 'settings'
]

controller = CookieController()

def is_sensitive_query(question):
    """Check if the user question contains sensitive information requests"""
    question_lower = question.lower()
    
    # Check for sensitive keywords
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in question_lower:
            return True
    
    # Check for patterns that might be asking about database structure
    sensitive_patterns = [
        r'what.*table.*have',
        r'show.*table',
        r'list.*table',
        r'describe.*table',
        r'what.*column',
        r'show.*column',
        r'list.*column',
        r'database.*structure',
        r'table.*structure',
        r'what.*field',
        r'show.*field',
        r'list.*field'
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, question_lower):
            return True
    
    return False

def check_cookie_login_and_expiry():
    """Checks for login in session_state or cookie, validates expiry using the .expires timestamp."""
    user_info = None

    # 1. Check Session State first (faster if already logged in in this session)
    if "user_info" in st.session_state and st.session_state.user_info is not None:
        user_info = st.session_state.user_info

    # 2. If not in session state, try to get from cookie
    if user_info is None:
        time.sleep(2)
        cookie_data = controller.get(COOKIE_NAME)
        print(f"Cookie value: {cookie_data}") # Debug print
        if cookie_data:
            try:
                user_info = {
                    **cookie_data.get("user_details", {}), # Restore minimal user details
                    "token_expires_at": cookie_data.get("token_expires_at") # Get expiry string
                }
                print("Found user_info in cookie.") # Debug print
                # Populate session state from cookie for persistence within this session
                st.session_state.user_info = user_info
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Invalid login cookie format: {e}")
                user_info = None # Ensure user_info is None on error
            except Exception as e:
                print(f"Error processing login cookie: {e}")
                user_info = None # Ensure user_info is None on error

    # 3. If user_info was found (either in state or cookie), perform expiry check
    if user_info is not None:
        expires_at_str = user_info.get("token_expires_at") # Get expiry string

        if expires_at_str:
            try:
                expires_datetime = parse_datetime_string(expires_at_str) # Parse expiry string
                now_utc = datetime.datetime.now(expires_datetime.tzinfo) if expires_datetime.tzinfo else datetime.datetime.now()

                if expires_datetime > now_utc:
                    return user_info # Return the valid user info
                else:
                    print("Token expired based on timestamp.") # Debug print
                    st.warning("Your session has expired. Please log in again.")
                    st.session_state.user_info = None # Clear expired state
                    return None # Return None as user is not valid
            except Exception as e:
                print(f"Error checking token expiry: {e}") # Debug print
                st.error("Error checking session validity. Please log in again.")
                st.session_state.user_info = None
                return None
        else:
            print("User info found, but expiry data missing.") # Debug print
            st.warning("Session information incomplete. Please log in again.")
            st.session_state.user_info = None
            return None

    return None

def get_database_schema_hash(db_path):
    """Generate a hash of the current database schema to detect changes."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table schemas
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name")
        schema_info = cursor.fetchall()
        
        # Get column information for all tables
        all_columns = []
        for table_name, _ in schema_info:
            if table_name != 'sqlite_sequence':  # Skip system tables
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                all_columns.extend([(table_name, col) for col in columns])
        
        conn.close()
        
        # Create a hash of the schema information
        schema_string = str(schema_info) + str(all_columns)
        return hashlib.md5(schema_string.encode()).hexdigest()
    
    except Exception as e:
        print(f"Error getting schema hash: {e}")
        return None

def has_schema_changed(db_path):
    """Check if the database schema has changed since last training."""
    current_hash = get_database_schema_hash(db_path)
    
    if current_hash is None:
        return True  # Assume changed if we can't get hash
    
    try:
        # Try to read the stored hash
        if os.path.exists(SCHEMA_HASH_FILE):
            with open(SCHEMA_HASH_FILE, 'r') as f:
                stored_hash = f.read().strip()
            return current_hash != stored_hash
        else:
            return True  # No stored hash means first time
    except Exception:
        return True  # Assume changed if we can't read stored hash

def save_schema_hash(db_path):
    """Save the current schema hash."""
    current_hash = get_database_schema_hash(db_path)
    if current_hash:
        try:
            with open(SCHEMA_HASH_FILE, 'w') as f:
                f.write(current_hash)
        except Exception as e:
            print(f"Error saving schema hash: {e}")

def clear_vector_database(vn):
    """Clear all existing training data from the vector database."""
    try:
        # Get all training data
        training_data = vn.get_training_data()
        print(f"Found {len(training_data)} training items to clear")
        
        # Remove each training item
        for item in training_data:
            if 'id' in item:
                try:
                    vn.remove_training_data(id=item['id'])
                except Exception as e:
                    print(f"Error removing training item {item['id']}: {e}")
        
        print("Vector database cleared successfully")
        return True
    except Exception as e:
        print(f"Error clearing vector database: {e}")
        return False

# --- Authentication and Expiration Check ---
logged_in_user_info = check_cookie_login_and_expiry()

if logged_in_user_info is None:
    print("Redirecting to login page.") # Debug print
    st.session_state.user_info = None # Ensure state is clear
    st.session_state.messages = [] # Clear chat history
    st.switch_page("streamlit_app.py") # Redirect to login page
    st.stop() # Stop execution of the rest of this page

# Vanna AI Custom Class
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

def get_sample_training_data():
    """Returns sample training data for sales database with cleaned column names"""
    return [
      {
        'question': 'What is the total total sales for May 2025?',
        'sql': "SELECT SUM([Total Sales]) AS total_total_sales FROM sales WHERE strftime('%Y-%m', [From_Date]) = '2025-05'"
    }]

@st.cache_resource
def setup_agent():
    """Sets up the Vanna AI agent with SQLite database."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return None, "OPENAI_API_KEY environment variable not set."
        
        # Get database URI
        db_uri = os.getenv('DB_URI')
        if not db_uri:
            return None, "DB_URI environment variable not set."
        
        db_path = db_uri.replace('sqlite:///', '')
        
        # Check if schema has changed
        schema_changed = has_schema_changed(db_path)
        
        # Initialize Vanna with OpenAI and ChromaDB
        vn = MyVanna(config={
            'api_key': openai_api_key,
            'model': 'gpt-4o'  # Using the same model as before
        })
        
        # Connect to SQLite database
        vn.connect_to_sqlite(db_path)
        
        # Check if we need to train or retrain the model
        training_data = vn.get_training_data()
        needs_training = len(training_data) == 0 or schema_changed
        
        if needs_training:
            if schema_changed:
                st.info("Database schema has changed. Clearing old training data and retraining Vanna AI...")
                # Clear existing training data
                clear_vector_database(vn)
            else:
                st.info("Training Vanna AI on your database schema and sample queries... This may take a moment.")
            
            # Train Vanna on DDL statements
            try:
                df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
                for ddl in df_ddl['sql'].to_list():
                    vn.train(ddl=ddl)
                
                # Add business documentation
                vn.train(documentation="""
                This is a sales database containing customer transaction data with the following key information:
                - Sales transactions with dates, amounts, and profit
                - Customer information and regions
                - Product categories and names
                - Time-based analysis capabilities for trends
                - Gross margin calculations
                - Regional and categorical breakdowns available
                
                Important: All column names have been cleaned and whitespace has been removed.
                """)
                
                # Train with sample question-SQL pairs
                sample_data = get_sample_training_data()
                for item in sample_data:
                    vn.train(question=item["question"], sql=item["sql"])
                
                # Save the current schema hash
                save_schema_hash(db_path)
                
                st.success("Vanna AI training completed with updated schema!")
            except Exception as train_error:
                print(f"Training error: {train_error}")
                # Continue even if training fails - the model might still work
        
        return vn, None
        
    except Exception as e:
        return None, f"Error setting up Vanna AI agent: {e}"

# Initialize the Vanna agent
vanna_agent, agent_setup_error = setup_agent()

# --- Streamlit UI (Chatbot) ---

st.title("SQL Database Chatbot")

# Display logged-in user info and Logout button in sidebar
user_name = st.session_state.user_info.get("Name", "User")
st.sidebar.write(f"Logged in as: **{user_name}**")

# Logout button
if st.sidebar.button("Logout"):
    st.session_state.user_info = None # Clear user info from session state
    st.session_state.messages = [] # Clear chat history
    controller.remove(COOKIE_NAME) # Delete the login cookie
    st.switch_page("streamlit_app.py") # Redirect back to the login page

st.write("Ask me something about the sales data:")

# Initialize chat history for this page
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Display the main response
            st.markdown(message["content"])
            
            # Display Vanna-generated chart if it exists
            if "vanna_chart" in message:
                st.plotly_chart(message["vanna_chart"], use_container_width=True)
                
            # Display data table if it exists
            if "dataframe" in message and not message["dataframe"].empty:
                st.write("📊 **Results:**")
                st.dataframe(message["dataframe"])
            
            # Display SQL query in expander if it exists
            if "sql_query" in message and message["sql_query"]:
                with st.expander("📝 View Generated SQL Query"):
                    st.code(message["sql_query"], language='sql')
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me something about the sales data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Check for sensitive queries first
    if is_sensitive_query(prompt):
        sensitive_response = "I don't have access to that type of information. Please ask questions about your sales data, such as sales trends, product performance, customer analysis, or revenue metrics."
        
        with st.chat_message("assistant"):
            st.write(sensitive_response)
        
        st.session_state.messages.append({"role": "assistant", "content": sensitive_response})
    
    # Check if agent setup was successful before attempting to use it
    elif agent_setup_error:
        st.warning("Vanna AI agent is not available due to setup error.")
        error_msg = f"Agent is not available due to setup error: {agent_setup_error}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    elif vanna_agent:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question and generating response..."):
                try:
                    # Use Vanna AI's built-in ask method which handles everything
                    # This returns SQL, DataFrame, chart, and explanation
                    result = vanna_agent.ask(prompt)
                    
                    # Vanna's ask method can return different types of results
                    # Let's handle them appropriately
                    if result is not None:
                        # Try to get the SQL query that was generated
                        try:
                            sql_query = vanna_agent.generate_sql(prompt)
                        except:
                            sql_query = None
                        
                        # Try to get the data
                        try:
                            if sql_query:
                                result_df = vanna_agent.run_sql(sql_query)
                            else:
                                result_df = pd.DataFrame()
                        except:
                            result_df = pd.DataFrame()
                        
                        # Try to get Vanna's generated chart
                        try:
                            # Vanna can generate charts automatically with get_plotly_figure
                            if not result_df.empty and sql_query:
                                vanna_chart = vanna_agent.get_plotly_figure(
                                    plotly_code=vanna_agent.generate_plotly_code(
                                        question=prompt,
                                        sql=sql_query,
                                        df=result_df
                                    ),
                                    df=result_df
                                )
                            else:
                                vanna_chart = None
                        except Exception as chart_error:
                            print(f"Chart generation error: {chart_error}")
                            vanna_chart = None
                        
                        # Generate a natural language response
                        if isinstance(result, str):
                            # If Vanna returned a string response, use it
                            answer = result
                        elif not result_df.empty:
                            # Generate response based on data
                            answer = f"I found {len(result_df)} result(s) for your query. "
                            
                            # Add specific insights based on the data
                            if len(result_df) == 1 and len(result_df.columns) >= 1:
                                # Single result - be specific
                                first_col = result_df.columns[0]
                                value = result_df.iloc[0][first_col]
                                if pd.api.types.is_numeric_dtype(result_df[first_col]):
                                    if isinstance(value, float):
                                        answer = f"The result is: {value:,.2f}"
                                    else:
                                        answer = f"The result is: {value:,}"
                                else:
                                    answer = f"The result is: {value}"
                            
                            elif len(result_df) <= 10:
                                # Small dataset - provide summary
                                numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                                if numeric_cols:
                                    total = result_df[numeric_cols[0]].sum()
                                    answer += f"The total for {numeric_cols[0]} is {total:,.2f}."
                        else:
                            answer = "No results were found for your query. The data might not contain information matching your request."
                        
                        # Display the main answer
                        st.write(answer)
                        
                        # Display Vanna-generated chart if available
                        chart_to_store = None
                        if vanna_chart:
                            st.plotly_chart(vanna_chart, use_container_width=True)
                            chart_to_store = vanna_chart
                        
                        # Display data table if available
                        if not result_df.empty:
                            st.write("📊 **Results:**")
                            st.dataframe(result_df)
                        
                        # Store the response in chat history
                        message_data = {
                            "role": "assistant", 
                            "content": answer,
                            "sql_query": sql_query,
                            "dataframe": result_df if not result_df.empty else pd.DataFrame()
                        }
                        
                        if chart_to_store:
                            message_data["vanna_chart"] = chart_to_store
                        
                        st.session_state.messages.append(message_data)
                        
                        # Display SQL in expander
                        if sql_query:
                            with st.expander("📝 View Generated SQL Query"):
                                st.code(sql_query, language='sql')
                    
                    else:
                        error_message = "I couldn't generate a response for your question. Please try rephrasing your question or ask about sales data, products, customers, or time-based analysis."
                        st.write(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

                except Exception as e:
                    # Check if this is a sensitive query that slipped through
                    if any(keyword in str(e).lower() for keyword in ['schema', 'table', 'sqlite_master']):
                        error_message = "I don't have access to that type of information. Please ask questions about your sales data instead."
                    else:
                        error_message = f"An error occurred while processing your request. Please try rephrasing your question."
                    
                    st.error(error_message)
                    # Add error message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        st.error("Vanna AI agent is not initialized.")

# Sidebar information
st.sidebar.info("""
**How SQL Agent works:**
1. **Training**: SQL Agent learns your database schema automatically
2. **Question Processing**: Your natural language question is analyzed
3. **SQL Generation**: AI generates appropriate SQL queries
4. **Execution**: Query runs on your SQLite database
5. **Results**: Data is returned with explanations and visualizations

**Sample Questions:**
- What is the total sales for June 2025?
- Show me average ticket sale by region and its total gross earning in May
- What is the sales count and gross earning by region and status show its trend.
- What is the gross margin of june?
- Show me sales trends over time

**Security Note:**
- Database structure and schema information is protected
- Only business data queries are allowed

**Powered by:**
- OpenAI GPT-4o
- ChromaDB Vector Store
- Generative AI RAG Framework
""")

# Add training data management in sidebar
if vanna_agent and st.sidebar.button("🔄 Force Retrain Model"):
    with st.spinner("Retraining Vanna AI..."):
        try:
            # Clear existing training data
            clear_vector_database(vanna_agent)
            
            # Retrain with schema
            df_ddl = vanna_agent.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
            for ddl in df_ddl['sql'].to_list():
                vanna_agent.train(ddl=ddl)
            
            # Add business documentation
            vanna_agent.train(documentation="""
            This is a sales database containing customer transaction data with cleaned column names.
            All column names have had leading/trailing whitespace removed for consistency.
            """)
            
            # Retrain with sample data
            sample_data = get_sample_training_data()
            for item in sample_data:
                vanna_agent.train(question=item["question"], sql=item["sql"])
            
            # Update schema hash
            db_path = os.getenv('DB_URI', '').replace('sqlite:///', '')
            save_schema_hash(db_path)
            
            st.sidebar.success("Model retrained successfully!")
        except Exception as e:
            st.sidebar.error(f"Retraining failed: {e}")

# Show current training data count and schema status
if vanna_agent:
    try:
        training_count = len(vanna_agent.get_training_data())
        st.sidebar.write(f"📚 Training data items: {training_count}")
        
        # Show schema status
        db_path = os.getenv('DB_URI', '').replace('sqlite:///', '')
        if has_schema_changed(db_path):
            st.sidebar.warning("⚠️ Schema may have changed")
        else:
            st.sidebar.success("✅ Schema is up to date")
    except:
        st.sidebar.write("📚 Training data: Unknown")