import time
import streamlit as st
import requests
import json
import os
import datetime
from dotenv import load_dotenv
from streamlit_cookies_controller import CookieController
from dateutil.parser import parse as parse_datetime_string

load_dotenv()

AUTH_API_URL = "http://posapi.iconnectgroup.com/Api/GetAuthToken"
LOGIN_API_URL = "http://posapi.iconnectgroup.com/Api/Pos/UserLogin"
AUTH_CODE = os.getenv('AUTH_CODE')
COOKIE_NAME = "login_state"

def get_auth_token(auth_code: str):
    """Calls the GetAuthToken API."""
    try:
        response = requests.post(AUTH_API_URL, data={
            "grant_type": "password",
            "AuthCode": auth_code
        })
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error (GetAuthToken): {e}")
        st.error("Failed to get authentication token.")
        return None
    except json.JSONDecodeError:
        print("API Error (GetAuthToken): Invalid JSON response.")
        st.error("Failed to get authentication token.")
        return None

def user_login(token: str, username: str, password: str):
    """Calls the UserLogin API using the obtained token."""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "UserName": username,
            "Password": password
        }
        response = requests.post(LOGIN_API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        if response is not None and response.status_code in [400, 401]:
            st.error("Login failed: Invalid username or password.")
        else:
            print(f"API Error (UserLogin): {e}")
            st.error("Login failed: Invalid username or password.")
        return None
    except json.JSONDecodeError:
        print("API Error (UserLogin): Invalid JSON response.")
        st.error("Login failed: Invalid username or password.")
        return None

controller = CookieController()

# --- Function to check cookie and restore session state ---
def check_cookie_login():
    """Checks for a login cookie and restores st.session_state if valid."""
    if "user_info" in st.session_state and st.session_state.user_info is not None:
        # Already logged in during this session, validate expiry
        user_info = st.session_state.user_info
        expires_at_str = user_info.get("token_expires_at") # Get expiry string

        if expires_at_str:
            try:
                expires_datetime = parse_datetime_string(expires_at_str) # Parse expiry string
                if expires_datetime > datetime.datetime.now(expires_datetime.tzinfo): # Compare with timezone-aware now() if expiry is tz-aware
                    print("Login validated from session state (expiry check passed).") # Debug print
                    return user_info
                else:
                    print("Session state token expired based on timestamp.") # Debug print
                    st.warning("Your session has expired. Please log in again.")
                    st.session_state.user_info = None # Clear expired state
                    controller.remove(COOKIE_NAME) # Also delete the cookie
                    return None
            except Exception as e:
                print(f"Error checking session state token expiry: {e}") # Debug print
                st.error("Error validating session. Please log in again.")
                st.session_state.user_info = None
                controller.remove(COOKIE_NAME)
                return None
        else:
            print("Session state user info incomplete (no expiry timestamp).") # Debug print
            st.warning("Session information incomplete. Please log in again.")
            st.session_state.user_info = None
            controller.remove(COOKIE_NAME)
            return None


    # If not in session state, try to get from cookie
    cookie_value = controller.get(COOKIE_NAME)

    if cookie_value:
        try:
            cookie_data = json.loads(cookie_value)
            expires_at_str = cookie_data.get("token_expires_at") # Get expiry string from cookie
            user_details = cookie_data.get("user_details")

            if expires_at_str and user_details:
                expires_datetime = parse_datetime_string(expires_at_str) # Parse expiry string

                if expires_datetime > datetime.datetime.now(expires_datetime.tzinfo): # Compare with current time
                    # Cookie is valid and not expired - Restore session state
                    st.session_state.user_info = {
                        **user_details,
                        "token_expires_at": expires_at_str # Store the expiry string
                        # access_token is not stored in cookie
                    }
                    print("Login restored from cookie.") # Debug print
                    return st.session_state.user_info
                else:
                    print("Login cookie expired based on timestamp.") # Debug print
                    st.warning("Your session has expired. Please log in again.")
                    controller.remove(COOKIE_NAME) # Clean up expired cookie
            else:
                print("Login cookie data incomplete during cookie check.") # Debug print
                controller.remove(COOKIE_NAME)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Invalid login cookie format: {e}") # Debug print
            controller.remove(COOKIE_NAME)
        except Exception as e:
            print(f"Error processing login cookie: {e}") # Debug print
            controller.remove(COOKIE_NAME)

    # If neither session state nor valid cookie worked
    print("No valid login found.") # Debug print
    return None


# --- Streamlit UI ---

st.title("App Login")

if not AUTH_CODE:
    st.error("Configuration Error: AUTH_CODE environment variable not set.")

logged_in_user_info = check_cookie_login()

if logged_in_user_info is not None:
    # If logged in (either from session or cookie), redirect
    st.info("Logged in. Redirecting...")
    st.switch_page("pages/chatbot_page.py")
else:
    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                # Step 1: Get Auth Token
                st.info("Attempting to get authentication token...")
                auth_response = get_auth_token(AUTH_CODE)

                if auth_response and auth_response.get("access_token"):
                    access_token = auth_response["access_token"]
                    expires_at_str = auth_response.get(".expires")
                    st.success("Authentication token obtained. Attempting user login...")

                    # Step 2: Login with Username and Password using the token
                    user_response = user_login(access_token, username, password)

                    if user_response and user_response.get("ContactID"):
                        # Successful login
                        st.session_state.user_info = {
                            **user_response, # User details
                            "token_expires_at": expires_at_str, # Token expiry string
                        }
                        cookie_data_to_store = {
                            "token_expires_at": expires_at_str,
                            "user_details": {
                                "ContactID": user_response.get("ContactID"),
                                "Name": user_response.get("Name"),
                                "Role": user_response.get("Role"),
                                # Add other minimal details needed for display/basic checks
                            }
                        }
                        expires_datetime = parse_datetime_string(expires_at_str)
                        controller.set(
                            name=COOKIE_NAME,
                            value=json.dumps(cookie_data_to_store),
                            max_age=86400
                        )
                        time.sleep(1) 
                        st.success(f"Welcome, {st.session_state.user_info.get('Name', 'User')}! Redirecting...")
                        st.switch_page("pages/chatbot_page.py")
                    else:
                        st.session_state.user_info = None # Ensure state is clear on failure
                else:
                    st.session_state.user_info = None # Ensure state is clear on failure
