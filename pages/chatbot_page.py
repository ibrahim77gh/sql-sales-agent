import time
import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from streamlit_cookies_controller import CookieController
from dateutil.parser import parse as parse_datetime_string
import json
import datetime

from utils import SYSTEM_PROMPT

load_dotenv()

COOKIE_NAME = "login_state"

controller = CookieController()

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
                # controller.remove(COOKIE_NAME)
                user_info = None # Ensure user_info is None on error
            except Exception as e:
                print(f"Error processing login cookie: {e}")
                # controller.remove(COOKIE_NAME)
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
                    # controller.remove(COOKIE_NAME) # Delete cookie
                    return None # Return None as user is not valid
            except Exception as e:
                print(f"Error checking token expiry: {e}") # Debug print
                st.error("Error checking session validity. Please log in again.")
                st.session_state.user_info = None
                # controller.remove(COOKIE_NAME)
                return None
        else:
            print("User info found, but expiry data missing.") # Debug print
            st.warning("Session information incomplete. Please log in again.")
            st.session_state.user_info = None
            # controller.remove(COOKIE_NAME)
            return None

    print("No valid login found.") # Debug print
    return None

# --- Authentication and Expiration Check ---
logged_in_user_info = check_cookie_login_and_expiry()

if logged_in_user_info is None:
    print("Redirecting to login page.") # Debug print
    st.session_state.user_info = None # Ensure state is clear
    st.session_state.messages = [] # Clear chat history
    st.switch_page("streamlit_app.py") # Redirect to login page
    st.stop() # Stop execution of the rest of this page

@st.cache_resource
def setup_agent():
    """Sets up the LLM, DB connection, and agent."""
    try:
        # Using a placeholder model name, replace if needed
        llm = init_chat_model(model="openai:gpt-4o-mini") # Example model
        db_uri = os.getenv('DB_URI')
        if not db_uri:
             return None, "DB_URI environment variable not set. Cannot connect to database."

        db = SQLDatabase.from_uri(db_uri)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        system_prompt = SYSTEM_PROMPT.format(
            dialect=db.dialect,
            top_k=5,
        )
        # Assuming create_react_agent takes prompt directly now
        agent = create_react_agent(
            llm,
            tools,
            prompt=system_prompt,
        )
        return agent, None
    except Exception as e:
        return None, f"Error setting up the agent or database connection: {e}"

agent, agent_setup_error = setup_agent()

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
# This should be initialized *after* the auth check
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me something about the sales data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if agent setup was successful before attempting to use it
    if agent_setup_error:
        st.warning("Agent is not available due to setup error.")
        # Add error message to chat history
        st.session_state.messages.append({"role": "assistant", "content": "Agent is not available due to setup error. Please fix the configuration."})
    elif agent:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                MAX_HISTORY_TO_SEND = 2
                recent_messages_to_send = st.session_state.messages[-MAX_HISTORY_TO_SEND:]
                # Prepare messages for the agent
                agent_input_messages = [
                    (HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
                    for msg in recent_messages_to_send
                ]

                final_answer_text = ""
                intermediate_steps_text = ""
                message_placeholder = st.empty() # Placeholder for streaming the assistant's final answer

                # Stream the agent's response
                try:
                    # agent.stream yields steps, each containing the current state's messages
                    # The last message in the step is usually the most recent one (tool call, observation, or final answer)
                    for step in agent.stream({"messages": agent_input_messages}, stream_mode="values"):
                        last_message: BaseMessage = step["messages"][-1]

                        # Capture ToolMessage output (this is the raw query result string)
                        if isinstance(last_message, ToolMessage) and last_message.name == 'sql_db_query':
                             # Attempt to get the actual query from tool_calls
                            query_to_show = "N/A (could not extract query)"
                            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                try:
                                    tool_call = last_message.tool_calls[0]
                                    if tool_call.type == 'tool_call' and tool_call.function:
                                        if tool_call.function.args:
                                            query_to_show = tool_call.function.args.get('query', 'N/A (query key not found)')
                                        else:
                                            query_to_show = 'N/A (no args found)'
                                except Exception as ex:
                                    query_to_show = f"Error extracting query: {ex}"
                                    pass # Fallback

                            intermediate_steps_text += f"\nTool: sql_db_query\nQuery: ```sql\n{query_to_show}\n```\n"
                            intermediate_steps_text += f"Observation:\n```\n{last_message.content}\n```\n"


                        # Capture the final AIMessage (this is the synthesized answer)
                        if isinstance(last_message, AIMessage):
                            # Append content for streaming effect if desired
                            # For simplicity here, we just capture the final content
                            final_answer_text = last_message.content
                            # Optionally update placeholder here for typing effect
                            message_placeholder.markdown(final_answer_text + "▌")


                    # After the stream finishes, display the final result
                    message_placeholder.markdown(final_answer_text)

                    # Optional: Display intermediate steps/thoughts
                    if intermediate_steps_text:
                        with st.expander("Show thinking process"):
                            st.text(intermediate_steps_text) # Use st.text for raw tool output/steps

                    # Store the final assistant message in history
                    st.session_state.messages.append({"role": "assistant", "content": final_answer_text})

                except Exception as e:
                    error_message = f"An error occurred while processing your request: {e}"
                    st.error(error_message)
                    # Add error message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


st.sidebar.info("""
**How it works:**
1. You logged in successfully on the previous page.
2. This page checks your login status.
3. The SQL agent is set up (once per app instance).
4. Your question goes to the agent.
5. The agent decides on a SQL query.
6. It runs the query using the `sql_db_query` tool (steps visible in "Show thinking process").
7. The agent synthesizes a final answer based on the result.
8. The app displays the agent's final synthesized answer.
""")

# Removed the built-in auth docs as they are not relevant to this custom implementation.