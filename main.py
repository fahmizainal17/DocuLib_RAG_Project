"""
Streamlit RAG Knowledge System with Role-Based Access, Supabase Storage, and st.secrets

This Streamlit app provides:
1. User authentication via Supabase (email/password login).
2. Role-based access control: documents can be tagged as "admin", "manager", or "worker".
3. A “Main” tab to upload `.txt`, `.pdf`, `.csv`, or `.xlsx` files, which are stored in Supabase Storage and indexed in FAISS via LangChain.
4. A “Q&A” tab to ask questions; the app retrieves relevant document chunks via LangChain FAISS and generates an answer using Gemini (`google.generativeai`).
5. A “Document Library” tab listing all uploaded documents filtered by the user’s role.
6. All secrets (Supabase, HF token, Gemini API key) come from `st.secrets`.

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
"""

import os
import tempfile
import streamlit as st
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.schema import Document
from PyPDF2 import PdfReader
import pandas as pd
import google.generativeai as genai

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

# ------------------------------------------------------------------------------
# Initialize clients and models
# ------------------------------------------------------------------------------

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set HF_HOME so SentenceTransformer picks up HF_TOKEN if needed
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
            vs = FAISS.from_texts([], embedding_model)
            vs.save_local(FAISS_DIR)
        st.session_state.vectorstore = vs
    except Exception as e:
        st.error(f"Error initializing FAISS vectorstore: {e}")
        vs = FAISS.from_texts([], embedding_model)
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

    Args:
        user_email (str): The authenticated user’s email.

    Returns:
        str: Role of the user ("admin", "manager", or "worker").
    """
    response = supabase.table("profiles").select("role").eq("email", user_email).single().execute()
    if response.error or response.data is None:
        return "worker"
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


def upload_file_to_supabase(file_buffer, filename: str, content_type: str) -> str:
    """
    Upload the raw file to Supabase Storage bucket "documents".

    Args:
        file_buffer: The file-like buffer from Streamlit uploader.
        filename (str): Name to store in Supabase.
        content_type (str): MIME type.

    Returns:
        str: Path relative to Supabase bucket or empty string on error.
    """
    res = supabase.storage.from_("documents").upload(filename, file_buffer, {"content-type": content_type})
    if res.error:
        st.error(f"Error uploading file: {res.error.message}")
        return ""
    return filename


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
    contexts = []
    for doc in docs:
        contexts.append(doc.page_content)
    return "\n---\n".join(contexts)


# ------------------------------------------------------------------------------
# Streamlit App: Authentication & Role-Based Tabs
# ------------------------------------------------------------------------------

from component import page_style  # Existing component for UI styling

# Apply custom styles from the component
page_style()

st.title("RAG Knowledge System")

# Initialize session_state keys
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None
if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = []  # Cache of documents metadata


def login():
    """
    Display login form and authenticate via Supabase.
    """
    st.subheader("🔑 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = supabase.auth.sign_in(email=email, password=password)
        if user.user:
            st.success("Login successful.")
            st.session_state["user_email"] = email
            st.session_state["user_role"] = get_user_role(email)
            # Fetch any previously uploaded documents for this user
            fetch_uploaded_documents()
            st.experimental_rerun()
        else:
            st.error("Invalid credentials. Please try again.")


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
    res = supabase.table("documents").select("*").or_(f"role.eq.{role},role.eq.worker").execute()
    if not res.error:
        st.session_state["uploaded_docs"] = res.data


# If not logged in, show login form
if not st.session_state["user_email"]:
    login()
    st.stop()

# Sidebar: show user info, logout button, and how to retrieve past uploads
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
        "Select a file", type=["txt", "pdf", "csv", "xlsx"], accept_multiple_files=False
    )
    assigned_role = st.selectbox("Assign access to role", ["worker", "manager", "admin"])

    if uploaded_file:
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size,
        }
        st.write(file_details)

        if st.button("Upload and Index"):
            # Step 1: Save raw file to Supabase Storage
            storage_path = f"{assigned_role}/{uploaded_file.name}"
            upload_file_to_supabase(uploaded_file, storage_path, uploaded_file.type)

            # Step 2: Read file content to text
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp_path = tmp.name

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
                # Step 3: Insert document metadata into Supabase 'documents' table
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
                if res.error:
                    st.error(f"Error inserting metadata: {res.error.message}")
                else:
                    doc_id = res.data[0]["id"]
                    st.success(f"Document metadata saved with ID {doc_id}.")

                    # Step 4: Chunk text and save chunks to 'document_chunks' table
                    chunks = chunk_text(text)
                    for idx, chunk_text in chunks:
                        supabase.table("document_chunks").insert(
                            {"doc_id": doc_id, "chunk_index": idx, "chunk_text": chunk_text}
                        ).execute()
                    st.info(f"Inserted {len(chunks)} chunks into Supabase table.")

                    # Step 5: Index document chunks into FAISS via LangChain
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
    # Use session_state cache of uploaded docs
    docs = st.session_state["uploaded_docs"]
    if not docs:
        st.info("No documents available for your role.")
    else:
        for doc in docs:
            st.markdown(f"**{doc['filename']}** (Role: {doc['role']})")
            if st.button(f"Download {doc['filename']}", key=f"download_{doc['id']}"):
                url = supabase.storage.from_("documents").get_public_url(doc["storage_path"])
                st.write(f"[Click here to download]({url})")
