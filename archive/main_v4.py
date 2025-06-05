"""
Streamlit RAG Knowledge System with Role-Based Access and st.secrets

This Streamlit app provides:
1. Role-based access control: documents can be tagged as "admin", "manager", or "worker".
2. A “Main” tab to upload `.txt`, `.pdf`, `.csv`, or `.xlsx` files, which are immediately indexed into FAISS via LangChain.
   - PDF text is extracted by PyPDF2 and manually chunked (~500 words).
3. A “Q&A” tab to ask questions; the app retrieves relevant document chunks via FAISS and generates an answer using Gemini (`google.generativeai`).
4. A “Document Library” tab listing all uploaded documents filtered by the user’s role.
5. All secrets (HF token and Gemini API key) come from `st.secrets` or can be defined directly below.
"""

import os
import tempfile
import streamlit as st
import google.generativeai as genai

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from PyPDF2 import PdfReader
import pandas as pd
import faiss
from component import page_style  # your custom CSS + sidebar styling

# ------------------------------------------------------------------------------
# Load secrets (or define directly)
# ------------------------------------------------------------------------------

# Hugging Face token for embeddings (can also be set via st.secrets)
HF_TOKEN = st.secrets.get("HF_TOKEN")

# Gemini API key (can also be set via st.secrets)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# ------------------------------------------------------------------------------
# Initialize models and vectorstore
# ------------------------------------------------------------------------------

# Ensure HuggingFace sees your token
os.environ["HF_HOME"] = HF_TOKEN

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Embedding model for FAISS via LangChain
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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
            # Load existing index
            vs = FAISS.load_local(
                FAISS_DIR, embedding_model, allow_dangerous_deserialization=True
            )
        else:
            # Create an empty FAISS index via from_texts([]), which sets up a valid docstore
            vs = FAISS.from_texts([], embedding_model)
            vs.save_local(FAISS_DIR)

        st.session_state.vectorstore = vs

    except Exception as e:
        st.error(f"Error initializing FAISS vectorstore: {e}")
        # Fallback to an empty index if load fails
        vs = FAISS.from_texts([], embedding_model)
        vs.save_local(FAISS_DIR)
        st.session_state.vectorstore = vs

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file using PyPDF2.

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
    """
    df = pd.read_csv(file_path)
    return " ".join(df.astype(str).apply(lambda row: " ".join(row.values), axis=1))


def extract_text_from_excel(file_path: str) -> str:
    """
    Convert Excel file to plain text by concatenating all cells.
    """
    df = pd.read_excel(file_path, engine="openpyxl")
    return " ".join(df.astype(str).apply(lambda row: " ".join(row.values), axis=1))


def chunk_text(text: str, max_words: int = 500) -> list[tuple[int, str]]:
    """
    Break the document text into chunks of approximately max_words words.
    Returns a list of (chunk_index, chunk_content).
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


def index_document_in_vectorstore(doc_id: int, text: str):
    """
    Add document chunks (manually chunked) to the FAISS vectorstore and persist.
    """
    chunks = chunk_text(text)
    documents = []
    for idx, chunk in chunks:
        metadata = {"doc_id": doc_id, "chunk_index": idx}
        documents.append(Document(page_content=chunk, metadata=metadata))

    if documents:
        st.session_state.vectorstore.add_documents(documents)
        st.session_state.vectorstore.save_local(FAISS_DIR)


def index_pdf_manual(doc_id: int, pdf_path: str):
    """
    Load the entire PDF as text, manually chunk it, embed, add to FAISS, and persist.
    """
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return
    index_document_in_vectorstore(doc_id, text)


def generate_answer_with_gemini(question: str, context: str) -> str:
    """
    Generate an answer using Gemini (`google.generativeai`) given question and context.
    """
    input_prompt = prompt.format(context=context, question=question)
    response = gemini_model.generate_content(input_prompt)
    return response.text


def search_vectorstore(question: str, top_k: int = 3) -> str:
    """
    Retrieve the top_k most relevant chunks from the FAISS vectorstore based on the question.
    """
    docs = st.session_state.vectorstore.similarity_search(question, k=top_k)
    contexts = [doc.page_content for doc in docs]
    return "\n---\n".join(contexts)

# ------------------------------------------------------------------------------
# Streamlit App: Role Selection & Tabs
# ------------------------------------------------------------------------------

# Apply custom styles from component.py
page_style()

st.title("RAG Knowledge System")

# Initialize session_state keys
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None
if "uploaded_docs" not in st.session_state:
    # Each entry: {"doc_id": int, "filename": str, "path": str, "role": str, "type": "pdf"/"csv"/"txt"/"xlsx"}
    st.session_state["uploaded_docs"] = []

# Role selection (no authentication; choose a role to filter)
st.sidebar.header("Select Your Role")
role_choice = st.sidebar.selectbox("Role:", ["worker", "manager", "admin"])
st.session_state["user_role"] = role_choice

# Main Tabs
tab1, tab2, tab3 = st.tabs(
    ["📤 Main (Upload)", "❓ Q&A", "📚 Document Library"]
)

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
        st.markdown(
            f"**File:** {uploaded_file.name}  |  **Size:** {uploaded_file.size // 1024} KB"
        )

        if st.button("Upload and Index"):
            # 1) Write uploaded bytes to a local temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_{uploaded_file.name}"
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp_path = tmp.name

            # 2) Determine file type and index appropriately
            if uploaded_file.type == "application/pdf":
                # Manually extract + chunk PDF text
                doc_id = len(st.session_state["uploaded_docs"]) + 1
                index_pdf_manual(doc_id, tmp_path)
                st.success("PDF file indexed into FAISS (manual chunking).")

                st.session_state["uploaded_docs"].append(
                    {
                        "doc_id": doc_id,
                        "filename": uploaded_file.name,
                        "path": tmp_path,
                        "role": assigned_role,
                        "type": "pdf",
                    }
                )
                st.success(
                    f"PDF '{uploaded_file.name}' uploaded for role '{assigned_role}'."
                )

            elif uploaded_file.type == "text/plain":
                text = open(tmp_path, "r", encoding="utf-8", errors="ignore").read()
                if text.strip():
                    doc_id = len(st.session_state["uploaded_docs"]) + 1
                    index_document_in_vectorstore(doc_id, text)
                    st.success("TXT file content indexed into FAISS.")
                    st.session_state["uploaded_docs"].append(
                        {
                            "doc_id": doc_id,
                            "filename": uploaded_file.name,
                            "path": tmp_path,
                            "role": assigned_role,
                            "type": "txt",
                        }
                    )
                    st.success(
                        f"TXT '{uploaded_file.name}' uploaded for role '{assigned_role}'."
                    )

            elif uploaded_file.type == "text/csv":
                text = extract_text_from_csv(tmp_path)
                if text.strip():
                    doc_id = len(st.session_state["uploaded_docs"]) + 1
                    index_document_in_vectorstore(doc_id, text)
                    st.success("CSV content indexed into FAISS.")
                    st.session_state["uploaded_docs"].append(
                        {
                            "doc_id": doc_id,
                            "filename": uploaded_file.name,
                            "path": tmp_path,
                            "role": assigned_role,
                            "type": "csv",
                        }
                    )
                    st.success(
                        f"CSV '{uploaded_file.name}' uploaded for role '{assigned_role}'."
                    )

            elif uploaded_file.type in [
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]:
                text = extract_text_from_excel(tmp_path)
                if text.strip():
                    doc_id = len(st.session_state["uploaded_docs"]) + 1
                    index_document_in_vectorstore(doc_id, text)
                    st.success("Excel content indexed into FAISS.")
                    st.session_state["uploaded_docs"].append(
                        {
                            "doc_id": doc_id,
                            "filename": uploaded_file.name,
                            "path": tmp_path,
                            "role": assigned_role,
                            "type": "xlsx",
                        }
                    )
                    st.success(
                        f"Excel '{uploaded_file.name}' uploaded for role '{assigned_role}'."
                    )
            else:
                st.error("Unsupported file type.")

# -------------------------------- Q&A ----------------------------------------
with tab2:
    st.header("Ask a Question")
    st.write("Ask a question and get answers based on your uploaded documents.")
    question = st.text_input("Enter your question here:")
    if st.button("Get Answer"):
        if not question:
            st.warning("⚠️ Please enter a question to get started.")
        else:
            context = search_vectorstore(question, top_k=3)
            if context.strip():
                answer = generate_answer_with_gemini(question, context)
                st.subheader("Answer:")
                st.write(answer)
                st.subheader("Context Used:")
                st.write(context)
            else:
                st.warning("No relevant context found. Try uploading more documents.")

# ------------------------- Document Library ----------------------------------
with tab3:
    st.header("Document Library")
    st.write("List of documents you have access to based on your role.")

    current_role = st.session_state["user_role"]
    docs = [
        doc
        for doc in st.session_state["uploaded_docs"]
        if doc["role"] == current_role
    ]

    if not docs:
        st.info("No documents available for your role.")
    else:
        for doc in docs:
            st.markdown(f"**{doc['filename']}** (Role: {doc['role']})")
            with open(doc["path"], "rb") as f:
                file_data = f.read()
            st.download_button(
                label=f"Download {doc['filename']}",
                data=file_data,
                file_name=doc["filename"],
            )
