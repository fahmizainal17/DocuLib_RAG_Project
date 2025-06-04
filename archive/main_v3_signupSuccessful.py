"""
Streamlit RAG Knowledge System with Role-Based Access, Supabase Storage, and st.secrets

This Streamlit app provides:
1. User authentication via Supabase (email/password login), restricted to an allow‐list of emails.
2. Role-based access control: documents can be tagged as "admin", "manager", or "worker".
3. A “Main” tab to upload `.txt`, `.pdf`, `.csv`, or `.xlsx` files, which are stored in Supabase Storage and indexed in FAISS via LangChain.
4. A “Q&A” tab to ask questions; the app retrieves relevant document chunks via LangChain FAISS and generates an answer using Gemini (`google.generativeai`).
5. A “Document Library” tab listing all uploaded documents filtered by the user’s role.
6. All secrets (Supabase URL/key, HF token, Gemini API key, and allowed signup emails) come from `st.secrets`.

Dependencies:
- streamlit
- supabase
- langchain-core
- langchain-community
- sentence_transformers
- PyPDF2
- pandas
- openpyxl
- google-generativeai
- faiss-cpu
"""

import os
import tempfile
import streamlit as st
import google.generativeai as genai
from supabase import create_client, Client
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import pandas as pd
import faiss
from component import page_style  # your custom CSS + sidebar styling


# ------------------------------------------------------------------------------
# Load secrets from Streamlit
# ------------------------------------------------------------------------------

# Supabase credentials from Streamlit secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Hugging Face token for embeddings
HF_TOKEN = st.secrets["HF_TOKEN"]

# Gemini API key from Streamlit secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Allow‐list of emails permitted to sign up
ALLOWED_SIGNUP_EMAILS = st.secrets["allowed_signup_emails"]


# ------------------------------------------------------------------------------
# Initialize clients and models
# ------------------------------------------------------------------------------

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set HF_HOME so HuggingFaceEmbeddings picks up HF_TOKEN if needed
os.environ["HF_HOME"] = HF_TOKEN

# Configure Gemini (Gemini-2.0-Flash)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Embedding model for FAISS via LangChain
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prompt template for answer generation
template = (
    "Based on the following information, provide a concise answer to the question:\n\n"
    "Information:\n{context}\n\n"
    "Question: {question}\n\n"
    "You must answer in the language of the question (e.g., English if the question is in English, Malay if in Malay).\n"
    "Answer concisely in 50-75 words:"
)
prompt = ChatPromptTemplate.from_template(template)

# Directory to persist the FAISS index
FAISS_DIR = "faiss_index"

# Initialize or load LangChain FAISS vectorstore in session_state
if "vectorstore" not in st.session_state:
    try:
        if os.path.isdir(FAISS_DIR):
            vs = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
        else:
            # Create an empty FAISS index manually to avoid errors on empty texts
            dummy_embedding = embedding_model.embed_query(" ")
            dim = len(dummy_embedding)
            index = faiss.IndexFlatL2(dim)
            vs = FAISS(embedding_model, index, {}, [])
            vs.save_local(FAISS_DIR)

        st.session_state.vectorstore = vs

    except Exception as e:
        st.error(f"Error initializing FAISS vectorstore: {e}")
        # Fallback to a brand‐new, empty index
        dummy_embedding = embedding_model.embed_query(" ")
        dim = len(dummy_embedding)
        index = faiss.IndexFlatL2(dim)
        vs = FAISS(embedding_model, index, {}, [])
        vs.save_local(FAISS_DIR)
        st.session_state.vectorstore = vs


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_csv(file_path: str) -> str:
    """
    Convert CSV file to plain text by concatenating all fields.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        str: Concatenated text of all rows.
    """
    df = pd.read_csv(file_path)
    return " ".join(df.astype(str).apply(lambda row: " ".join(row.values), axis=1))


def extract_text_from_excel(file_path: str) -> str:
    """
    Convert Excel file to plain text by concatenating all cells.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        str: Concatenated text of all cells.
    """
    df = pd.read_excel(file_path, engine="openpyxl")
    return " ".join(df.astype(str).apply(lambda row: " ".join(row.values), axis=1))


def get_user_role(user_email: str) -> str:
    """
    Fetch the user’s role from Supabase 'profiles' table based on email.
    If no row is found (or an error occurs), default to "worker".
    """
    try:
        response = (
            supabase
            .table("profiles")
            .select("role")
            .eq("email", user_email)
            .maybe_single()   # returns one row or None
            .execute()
        )
    except Exception:
        # If anything goes wrong (network, missing table, etc.), return "worker"
        return "worker"

    # If Supabase returned no row, data will be None
    if not response or response.data is None:
        return "worker"

    # Otherwise, response.data is a dict like {"role": "admin"}
    return response.data.get("role", "worker")


def chunk_text(text: str, max_words: int = 500) -> list[tuple[int, str]]:
    """
    Break the document text into chunks of approximately max_words words.

    Args:
        text (str): Entire text content of the document.
        max_words (int): Maximum approximate number of words per chunk.

    Returns:
        list[tuple[int, str]]: List of tuples (chunk_index, chunk_text).
    """
    sentences = text.split(". ")
    chunks = []
    current = ""
    chunk_index = 0
    for sent in sentences:
        if len(current.split()) + len(sent.split()) > max_words:
            chunks.append((chunk_index, current.strip()))
            chunk_index += 1
            current = sent + ". "
        else:
            current += sent + ". "
    if current.strip():
        chunks.append((chunk_index, current.strip()))
    return chunks


def upload_file_to_supabase(local_path: str, storage_filename: str, content_type: str) -> str:
    """
    Upload a local file (given by its filesystem path) to Supabase Storage bucket "documents".

    Args:
        local_path (str): Path on disk to the file we want to upload.
        storage_filename (str): The destination path/name (within the bucket).
        content_type (str): MIME type of the file.

    Returns:
        str: The same storage_filename on success, or empty string on error.
    """
    try:
        with open(local_path, "rb") as f:
            res = supabase.storage.from_("documents").upload(storage_filename, f, {"content-type": content_type})
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return ""

    # In supabase-py v2, if upload fails, an exception is thrown, or res.data will be None
    if not res or res.data is None:
        st.error("Error uploading file: no data returned.")
        return ""

    return storage_filename


def index_document_in_vectorstore(doc_id: int, text: str):
    """
    Add document chunks to the LangChain FAISS vectorstore and persist.

    Args:
        doc_id (int): Unique identifier for the document.
        text (str): Entire text content of the document.
    """
    chunks = chunk_text(text)
    documents = []
    for idx, chunk_text in chunks:
        metadata = {"doc_id": doc_id, "chunk_index": idx}
        documents.append(Document(page_content=chunk_text, metadata=metadata))

    if documents:
        st.session_state.vectorstore.add_documents(documents)
        st.session_state.vectorstore.save_local(FAISS_DIR)


def generate_answer_with_gemini(question: str, context: str) -> str:
    """
    Generate an answer using Gemini (`google.generativeai`) given question and context.

    Args:
        question (str): User’s question.
        context (str): Contextual text retrieved via FAISS.

    Returns:
        str: Generated answer from Gemini.
    """
    input_prompt = prompt.format(context=context, question=question)
    response = gemini_model.generate_content(input_prompt)
    return response.text


def search_vectorstore(question: str, top_k: int = 3) -> str:
    """
    Retrieve the top_k most relevant chunks from the FAISS vectorstore based on the question.

    Args:
        question (str): The user’s question.
        top_k (int): Number of top results.

    Returns:
        str: Concatenated chunk texts as context.
    """
    docs = st.session_state.vectorstore.similarity_search(question, k=top_k)
    contexts = [doc.page_content for doc in docs]
    return "\n---\n".join(contexts)


# ------------------------------------------------------------------------------
# Streamlit App: Authentication & Role-Based Tabs
# ------------------------------------------------------------------------------

# Apply custom styles from component.py
page_style()

st.title("RAG Knowledge System")

# Initialize session_state keys
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None
if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = []  # Cache of documents metadata


def login_or_signup():
    """
    Display two tabs: ‘Login’ and ‘Sign Up’.
    Only emails in ALLOWED_SIGNUP_EMAILS are allowed to register.
    """
    st.subheader("🔑 Access DocuLib")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    # ─── LOGIN TAB ───
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
            elif user_obj is None or not user_obj.email:
                st.error("❌ Login failed (no user returned). Please try again.")
            else:
                st.success("✅ Login successful!")
                st.balloons()
                st.session_state["user_email"] = user_obj.email
                st.session_state["user_role"] = get_user_role(user_obj.email)
                fetch_uploaded_documents()
                st.rerun()

    # ─── SIGN UP TAB ───
    with tab_signup:
        email_s = st.text_input("New Email", key="signup_email")
        password_s = st.text_input("New Password", type="password", key="signup_password")
        if st.button("Sign Up", key="signup_button"):
            if email_s not in ALLOWED_SIGNUP_EMAILS:
                st.error(
                    "❌ You are not allowed to sign up with that email.\n"
                    "Only the following addresses may register:\n\n"
                    f"{', '.join(ALLOWED_SIGNUP_EMAILS)}\n\n"
                    "Please contact the administrator if your email is missing."
                )
            else:
                try:
                    signup_resp = supabase.auth.sign_up({
                        "email": email_s,
                        "password": password_s
                    })
                except Exception as e:
                    msg = str(e).lower()
                    if "you can only request this after" in msg:
                        st.error("⚠️ Please wait approximately 1 minute before trying to sign up again.")
                    else:
                        st.error(f"❌ Sign-up failed: {e}")
                    return

                err = getattr(signup_resp, "error", None)
                user_obj = getattr(signup_resp, "user", None)

                if err:
                    st.error(f"❌ Sign-up failed: {err.message if hasattr(err, 'message') else err}")
                elif user_obj is None:
                    st.success(
                        "✅ Sign-up successful! Please check your inbox for a confirmation email (if enabled)."
                    )
                    st.info("After verifying your email, please log in on the Login tab.")
                else:
                    st.success("✅ Sign-up successful! You can now log in.")


def logout():
    """
    Log the user out of Supabase and clear session state.
    """
    supabase.auth.sign_out()
    st.session_state["user_email"] = None
    st.session_state["user_role"] = None
    st.stop()


def fetch_uploaded_documents():
    """
    Retrieve all documents from Supabase 'documents' table for the logged-in user.
    Populate session_state['uploaded_docs'].
    """
    role = st.session_state["user_role"]
    try:
        res = (
            supabase
            .table("documents")
            .select("*")
            .or_(f"role.eq.{role},role.eq.worker")
            .execute()
        )
    except Exception:
        # If the table doesn't exist or something goes wrong, return empty
        st.session_state["uploaded_docs"] = []
        return

    docs = res.data or []
    st.session_state["uploaded_docs"] = docs


# If nobody is logged in yet, show the login/sign-up UI and stop further rendering
if not st.session_state.get("user_email"):
    login_or_signup()
    st.stop()


# Sidebar: user info + logout + previously uploaded docs
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state['user_email']}")
    st.markdown(f"**Role:** {st.session_state['user_role']}")
    if st.button("Logout"):
        logout()
    st.markdown("---")
    st.markdown("### 📂 Previously Uploaded Documents")
    if st.session_state["uploaded_docs"]:
        for doc in st.session_state["uploaded_docs"]:
            st.markdown(f"- {doc['filename']} (Role: {doc['role']})")
    else:
        st.markdown("No documents found.")
    st.markdown("---")
    st.markdown("ℹ️ Documents are fetched from Supabase on login and cached in session state.")


# Main Tabs
tab1, tab2, tab3 = st.tabs(["📤 Main (Upload)", "❓ Q&A", "📚 Document Library"])


# ---------------------------- Main (Upload) -----------------------------------
with tab1:
    st.header("Upload Documents")
    st.write("Upload `.txt`, `.pdf`, `.csv`, or `.xlsx` files and assign role-based access.")

    uploaded_file = st.file_uploader(
        "Select a file",
        type=["txt", "pdf", "csv", "xlsx"],
        accept_multiple_files=False,
    )
    assigned_role = st.selectbox("Assign access to role", ["worker", "manager", "admin"])

    if uploaded_file:
        # (Optional) Show filename / size summary in a simple line:
        st.markdown(f"**File:** {uploaded_file.name}  |  **Size:** {uploaded_file.size // 1024} KB")

        if st.button("Upload and Index"):
            # Step 1: Write the uploaded bytes to a local temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp_path = tmp.name

            # Step 2: Upload that temp file to Supabase
            storage_path = f"{assigned_role}/{uploaded_file.name}"
            saved = upload_file_to_supabase(tmp_path, storage_path, uploaded_file.type)
            if not saved:
                st.stop()

            # Step 3: Extract text from the local temp file for indexing
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(tmp_path)
            elif uploaded_file.type == "text/plain":
                text = open(tmp_path, "r", encoding="utf-8", errors="ignore").read()
            elif uploaded_file.type == "text/csv":
                text = extract_text_from_csv(tmp_path)
            elif uploaded_file.type in [
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]:
                text = extract_text_from_excel(tmp_path)
            else:
                st.error("Unsupported file type.")
                text = ""

            if text:
                # Step 4: Insert document metadata into Supabase 'documents' table
                try:
                    res = (
                        supabase.table("documents")
                        .insert(
                            {
                                "filename": uploaded_file.name,
                                "storage_path": storage_path,
                                "role": assigned_role,
                                "uploaded_by": st.session_state["user_email"],
                            }
                        )
                        .execute()
                    )
                except Exception as e:
                    st.error(f"Error inserting metadata: {e}")
                    text = ""
                else:
                    if not res or res.data is None:
                        st.error("Error inserting metadata (no data returned).")
                        text = ""
                    else:
                        doc_id = res.data[0].get("id")
                        st.success(f"Document metadata saved with ID {doc_id}.")

                        # Step 5: Chunk text and save chunks to 'document_chunks' table
                        chunks = chunk_text(text)
                        for idx, chunk_text in chunks:
                            try:
                                supabase.table("document_chunks").insert(
                                    {
                                        "doc_id": doc_id,
                                        "chunk_index": idx,
                                        "chunk_text": chunk_text,
                                    }
                                ).execute()
                            except Exception:
                                pass
                        st.info(f"Inserted {len(chunks)} chunks into Supabase table.")

                        # Step 6: Index document chunks into FAISS via LangChain
                        index_document_in_vectorstore(doc_id, text)
                        st.success("Document indexed in FAISS.")

                        # Update session_state cache of uploaded docs
                        fetch_uploaded_documents()


# -------------------------------- Q&A ----------------------------------------
with tab2:
    st.header("Ask a Question")
    st.write("Ask a question and get answers based on your uploaded documents.")
    question = st.text_input("Enter your question here:")
    if st.button("Get Answer"):
        if question:
            context = search_vectorstore(question, top_k=3)
            if context:
                answer = generate_answer_with_gemini(question, context)
                st.subheader("Answer:")
                st.write(answer)
                st.subheader("Context Used:")
                st.write(context)
            else:
                st.warning("No relevant context found. Try uploading more documents.")
        else:
            st.warning("⚠️ Please enter a question to get started.")


# ------------------------- Document Library ----------------------------------
with tab3:
    st.header("Document Library")
    st.write("List of documents you have access to based on your role.")
    docs = st.session_state["uploaded_docs"]
    if not docs:
        st.info("No documents available for your role.")
    else:
        for doc in docs:
            st.markdown(f"**{doc['filename']}** (Role: {doc['role']})")
            if st.button(f"Download {doc['filename']}", key=f"download_{doc['id']}"):
                url = supabase.storage.from_("documents").get_public_url(doc["storage_path"])
                st.write(f"[Click here to download]({url})")
