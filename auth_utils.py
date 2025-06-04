import streamlit as st
from supabase import create_client, Client

# ────────────────────────────────────────────────────────────────────────────────
# 1) Initialize Supabase client (reuse the same URL/KEY as in main.py)
# ────────────────────────────────────────────────────────────────────────────────

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ────────────────────────────────────────────────────────────────────────────────
# 2) get_user_role: query the `profiles` table and return “admin/manager/worker”
# ────────────────────────────────────────────────────────────────────────────────
def get_user_role(user_email: str) -> str:
    """
    Fetch the user’s role from Supabase `profiles` table based on email.
    If no row is found (or an error occurs), default to "worker".
    """
    try:
        response = (
            supabase
            .table("profiles")
            .select("role")
            .eq("email", user_email)
            .maybe_single()
            .execute()
        )
    except Exception:
        return "worker"

    if not response or response.data is None:
        return "worker"

    return response.data.get("role", "worker")


# ────────────────────────────────────────────────────────────────────────────────
# 3) fetch_uploaded_documents: get all rows from `documents` visible to current role
# ────────────────────────────────────────────────────────────────────────────────
def fetch_uploaded_documents() -> None:
    """
    Retrieve all documents from Supabase `documents` table for the logged-in user.
    Populates st.session_state["uploaded_docs"] with a list of rows or an empty list.
    """
    role = st.session_state.get("user_role", "worker")
    try:
        res = (
            supabase
            .table("documents")
            .select("*")
            .or_(f"role.eq.{role},role.eq.worker")
            .execute()
        )
    except Exception:
        st.session_state["uploaded_docs"] = []
        return

    st.session_state["uploaded_docs"] = res.data or []


# ────────────────────────────────────────────────────────────────────────────────
# 4) login_or_signup: the two‐tab UI that handles Supabase Auth
# ────────────────────────────────────────────────────────────────────────────────
def login_or_signup(allowed_emails: list[str]) -> None:
    """
    Display “Login” / “Sign Up” tabs.  Only emails in `allowed_emails` may register.
    On a successful login, store `user_email` + `user_role` in session_state,
    then balloon and rerun to show the main UI.
    """
    st.subheader("🔑 Access DocuLib")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    # ─── LOGIN ───
    with tab_login:
        email_l = st.text_input("Email", key="login_email")
        password_l = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_button"):
            try:
                login_resp = supabase.auth.sign_in_with_password({
                    "email": email_l,
                    "password": password_l
                })
            except Exception as e:
                st.error(f"❌ Login error: {e}")
                return

            err = getattr(login_resp, "error", None)
            user_obj = getattr(login_resp, "user", None)

            if err:
                st.error(f"❌ Login failed: {err.message if hasattr(err, 'message') else err}")
                return
            if user_obj is None or not user_obj.email:
                st.error("❌ Login failed (no user returned). Please try again.")
                return

            # ── Success: set session_state + show balloons + fetch docs + rerun ──
            st.success("✅ Login successful!")
            st.balloons()
            st.session_state["user_email"] = user_obj.email
            st.session_state["user_role"] = get_user_role(user_obj.email)
            fetch_uploaded_documents()
            st.rerun()

    # ─── SIGN UP ───
    with tab_signup:
        email_s = st.text_input("New Email", key="signup_email")
        password_s = st.text_input("New Password", type="password", key="signup_password")

        if st.button("Sign Up", key="signup_button"):
            if email_s not in allowed_emails:
                st.error(
                    "❌ You are not allowed to sign up with that email.\n"
                    "Only these addresses may register:\n\n"
                    f"{', '.join(allowed_emails)}\n\n"
                    "Please contact the administrator if your email is missing."
                )
                return

            try:
                signup_resp = supabase.auth.sign_up({
                    "email": email_s,
                    "password": password_s
                })
            except Exception as e:
                msg = str(e).lower()
                if "you can only request this after" in msg:
                    st.error("⚠️ Please wait ~1 minute before trying to sign up again.")
                else:
                    st.error(f"❌ Sign‐up failed: {e}")
                return

            err = getattr(signup_resp, "error", None)
            user_obj = getattr(signup_resp, "user", None)

            if err:
                st.error(f"❌ Sign‐up failed: {err.message if hasattr(err, 'message') else err}")
            elif user_obj is None:
                st.success("✅ Sign‐up initiated! Check your email for confirmation.")
                st.info("After verifying, please return to “Login” tab to sign in.")
            else:
                st.success("✅ Sign‐up successful! You can now log in.")
                

# ────────────────────────────────────────────────────────────────────────────────
# 5) logout(): clear supabase session and session_state keys
# ────────────────────────────────────────────────────────────────────────────────
def logout() -> None:
    supabase.auth.sign_out()
    st.session_state["user_email"] = None
    st.session_state["user_role"] = None
    st.stop()
