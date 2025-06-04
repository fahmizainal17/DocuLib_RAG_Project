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

# main.py

import os
import tempfile
import streamlit as st
import google.generativeai as genai

from supabase import Client
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import pandas as pd
import faiss

from component import page_style  # your custom CSS + sidebar styling

# ────────────────────────────────────────────────────────────────────────────────
# 1) Import everything from auth_utils instead of redefining it here
# ────────────────────────────────────────────────────────────────────────────────
from auth_utils import (
    login_or_signup,
    logout,
    fetch_uploaded_documents,
    get_user_role,
)

# ────────────────────────────────────────────────────────────────────────────────
# 2) Load "static" secrets (HF_TOKEN, GEMINI_API_KEY, ALLOWED_SIGNUP_EMAILS)
# ────────────────────────────────────────────────────────────────────────────────
HF_TOKEN = st.secrets["HF_TOKEN"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
ALLOWED_SIGNUP_EMAILS = st.secrets["allowed_signup_emails"]

# ────────────────────────────────────────────────────────────────────────────────
# 3) Initialize Supabase client (but since auth_utils already did that, you can
#    if desired import supabase from auth_utils, or just re-create here. We'll
#    not re-create a second client—it's already in auth_utils.)
# ────────────────────────────────────────────────────────────────────────────────
# (Note: We do NOT need to create another supabase client here because auth_utils
#  already did. If you really want supabase available in main.py, you can do:)
# from auth_utils import supabase


# ────────────────────────────────────────────────────────────────────────────────
# 4) Configure Gemini & embeddings exactly as before
# ────────────────────────────────────────────────────────────────────────────────
os.environ["HF_HOME"] = HF_TOKEN
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
prompt = ChatPromptTemplate.from_template(
    "Based on the following information, provide a concise answer to the question:\n\n"
    "Information:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer concisely in 50-75 words."
)

FAISS_DIR = "faiss_index"
if "vectorstore" not in st.session_state:
    try:
        if os.path.isdir(FAISS_DIR):
            vs = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
        else:
            # Create an empty FAISS index manually
            dummy_embedding = embedding_model.embed_query(" ")
            dim = len(dummy_embedding)
            index = faiss.IndexFlatL2(dim)
            vs = FAISS(embedding_model, index, {}, [])
            vs.save_local(FAISS_DIR)

        st.session_state.vectorstore = vs

    except Exception as e:
        st.error(f"Error initializing FAISS: {e}")
        dummy_embedding = embedding_model.embed_query(" ")
        dim = len(dummy_embedding)
        index = faiss.IndexFlatL2(dim)
        vs = FAISS(embedding_model, index, {}, [])
        vs.save_local(FAISS_DIR)
        st.session_state.vectorstore = vs


# ────────────────────────────────────────────────────────────────────────────────
# 5) Everything else in main.py remains exactly as you had it, except the login
#    block is replaced by a single import‐and‐call.
# ────────────────────────────────────────────────────────────────────────────────

# a) Apply CSS
page_style()

st.title("RAG Knowledge System")

# b) Ensure session_state keys exist
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None
if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = []


# ────────────────────────────────────────────────────────────────────────────────
# 6) Instead of embedding login/signup code here, simply call the imported helper
# ────────────────────────────────────────────────────────────────────────────────
if not st.session_state.get("user_email"):
    login_or_signup(ALLOWED_SIGNUP_EMAILS)
    st.stop()


# ────────────────────────────────────────────────────────────────────────────────
# 7) Sidebar: user info, logout, and “Previously Uploaded Documents”
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state['user_email']}")
    st.markdown(f"**Role:** {st.session_state['user_role']}")
    if st.button("Logout"):
        logout()

    st.markdown("---")
    st.markdown("### 📂 Previously Uploaded Documents")
    if st.session_state["uploaded_docs"]:
        for doc in st.session_state["uploaded_docs"]:
            st.markdown(f"- {doc['filename']}  (Role: {doc['role']})")
    else:
        st.markdown("No documents found.")
    st.markdown("---")
    st.markdown("ℹ️ Documents are fetched on login and cached in session state.")


# ────────────────────────────────────────────────────────────────────────────────
# 8) The “three‐tab” interface (Main, Q&A, Document Library) remains untouched
# ────────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Main (Upload)", "❓ Q&A", "📚 Document Library"])

# ---- Main (Upload) ----
with tab1:
    st.header("Upload Documents")
    st.write("Upload `.txt`, `.pdf`, `.csv`, or `.xlsx` files and assign role-based access.")

    uploaded_file = st.file_uploader(
        "Select a file", type=["txt", "pdf", "csv", "xlsx"], accept_multiple_files=False
    )
    assigned_role = st.selectbox("Assign access to role", ["worker", "manager", "admin"])

    if uploaded_file:
        st.markdown(f"**File:** {uploaded_file.name}  |  **Size:** {uploaded_file.size // 1024} KB")

        if st.button("Upload and Index"):
            # (1) Write bytes to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp_path = tmp.name

            # (2) Upload the temp file to Supabase
            from auth_utils import upload_file_to_supabase  # import helper
            storage_path = f"{assigned_role}/{uploaded_file.name}"
            saved = upload_file_to_supabase(tmp_path, storage_path, uploaded_file.type)
            if not saved:
                st.stop()

            # (3) Extract text for indexing
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
                try:
                    res = (
                        supabase.table("documents")
                        .insert({
                            "filename": uploaded_file.name,
                            "storage_path": storage_path,
                            "role": assigned_role,
                            "uploaded_by": st.session_state["user_email"],
                        })
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

                        # (4) Insert chunks into `document_chunks`
                        chunks = chunk_text(text)
                        for idx, chunk_text in chunks:
                            try:
                                supabase.table("document_chunks").insert({
                                    "doc_id": doc_id,
                                    "chunk_index": idx,
                                    "chunk_text": chunk_text,
                                }).execute()
                            except Exception:
                                pass
                        st.info(f"Inserted {len(chunks)} chunks into Supabase.")

                        # (5) Index into FAISS
                        index_document_in_vectorstore(doc_id, text)
                        st.success("Document indexed in FAISS.")

                        # (6) Refresh session_state cache
                        fetch_uploaded_documents()


# ---- Q&A ----
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


# ---- Document Library ----
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
