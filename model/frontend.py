import streamlit as st
import requests
#-------------------------------------------------------------------------------------------------------------------------------------------
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Cancer Care AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#state dict
# ---------------------------------------------------------------------------
def state():
    config={
        "token": None,
        "user_id": None,
        "username": None,
        "full_name": None,
        "page": "login",
        "active_conv_id": None,
        "chat_messages": [],
    }
    for k,v in config.items():
        if k not in st.session_state:
            st.session_state[k]=v

state()

def update_state(input):
    for k,v in input.items():
        if k not in st.session_state:
            return("you are tying to update state variable which is not initiated")
        st.session_state[k]=v

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

def api_post(endpoint, json=None, files=None, auth=True):
    headers = api_headers() if auth else {}
    try:
        r = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=json,
            files=files,
            headers=headers,
            timeout=60,
        )
        if r.status_code == 401:
            st.error("Your session has expired. Please log in again.")
            st.stop() # Stops the rest of the app from running
        elif r.status_code == 200 & st.session_state.page in [""]:#add pages to show login message
            st.success("Data fetched successfully!")
        return r

    except requests.ConnectionError:
        st.error(" Cannot connect to backend. Make sure uvicorn is running on port 8000.")
        return None
    

def api_get(endpoint):
    try:
        r = requests.get(
            f"{BACKEND_URL}{endpoint}",
            headers=api_headers(),
            timeout=30,
        )
        return r
    except requests.ConnectionError:
        st.error(f" Cannot connect to backend error while connecting to {endpoint}")
        return None


def api_delete(endpoint):
    try:
        r = requests.delete(
            f"{BACKEND_URL}{endpoint}",
            headers=api_headers(),
            timeout=30,
        )
        return r
    except requests.ConnectionError:
        st.error(f" Cannot connect to backend error while connecting to {endpoint}")
        return None

def safe_json(r):
    """Safely parse JSON from a response; return {} on failure."""
    try:
        return r.json()
    except Exception:
        return {}


def logout():
    for key in ["token", "user_id", "username", "full_name", "active_conv_id", "chat_messages"]:
        st.session_state[key] = None
    st.session_state["chat_messages"] = []
    st.session_state["page"] = "login"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def page_login():
    # ... (your existing markdown and form code) ...
    st.markdown('<div class="gradient-title">Welcome Back 👋</div>', unsafe_allow_html=True)
    st.markdown("Sign in to your Cancer Care AI account.")
    st.markdown("")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="your_username")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In →", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Please fill in all fields.")
            return
        
        r = api_post("/auth/login", json={"username": username, "password": password}, auth=False)
        
        if r and r.status_code == 200:
            data = r.json()
            # Update session state
            st.session_state.token = data["access_token"]
            st.session_state.user_id = data["user_id"]
            st.session_state.username = data["username"]
            st.session_state.full_name = data["full_name"]
            st.session_state.page = "chat"
            
            st.success("✅ Login successful!")
            st.rerun()  # <--- CRITICAL: This refreshes the app to show the chat page
        else:
            error_msg = safe_json(r).get('detail', 'Login failed')
            st.error(f"❌ {error_msg}")

def main():
    # Routing Logic
    if st.session_state.token is None or st.session_state.page == "login":
        page_login()
    elif st.session_state.page == "chat":
        st.write(f"Welcome, {st.session_state.full_name}! This is the Chat Page.")
        if st.button("Logout"):
            logout()
            st.rerun()

if __name__ == "__main__":
    main()