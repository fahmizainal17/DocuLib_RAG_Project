"""
Streamlit RAG Knowledge System with Role-Based Access and Password Authentication

This Streamlit app provides:
1. Role-based access control with password authentication for "admin", "manager", or "worker" roles.
   - Admin: Access all files (admin, manager, worker).
   - Manager: Access manager and worker files.
   - Worker: Access worker files only.
   - Passwords stored in st.secrets (ADMIN_PASSWORD, MANAGER_PASSWORD, WORKER_PASSWORD).
2. A "Main" tab for role login and to upload .txt, .pdf, .csv, .xlsx, .pptx files, or input website/YouTube URLs, indexed into FAISS via LangChain.
   - PDF text is extracted by PyPDFLoader and manually chunked (~500 words).
   - Website content is extracted using AsyncHtmlLoader and Html2TextTransformer.
   - YouTube videos are transcribed using yt-dlp and OpenAI Whisper.
   - PowerPoint files are extracted using UnstructuredPowerPointLoader.
3. A "Q&A" tab to ask questions, retrieving relevant document chunks via FAISS and generating answers using Gemini (google.generativeai).
4. A "Document Library" tab listing uploaded documents filtered by user role, displayed in a table with file type.
5. All secrets (HF token, Gemini API key, OpenAI API key, passwords) come from st.secrets.
"""

import os

# Set environment variables *before* importing any Hugging Face / LangChain modules
os.environ["USER_AGENT"] = "DocuLibRAG/1.0 (fahmizainals9@gmail.com)"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------

import shutil
import tempfile
import streamlit as st
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import UnstructuredPowerPointLoader
import pandas as pd
import faiss
import nest_asyncio
from openai import OpenAI
import yt_dlp
from component import page_style 

# Apply nest_asyncio for AsyncHtmlLoader compatibility
nest_asyncio.apply()

MAX_FILE_SIZE_MB = 10

# ------------------------------------------------------------------------------
# Load secrets
# ------------------------------------------------------------------------------

HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("Hugging Face token not found in st.secrets. Please set HF_TOKEN in .streamlit/secrets.toml.")
    st.stop()

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in st.secrets. Please set GEMINI_API_KEY in .streamlit/secrets.toml.")
    st.stop()

OPENAI_TRANSCRIPTION_API_KEY = st.secrets.get("OPENAI_TRANSCRIPTION_API_KEY")
if not OPENAI_TRANSCRIPTION_API_KEY:
    st.error("OpenAI API key not found in st.secrets. Please set OPENAI_TRANSCRIPTION_API_KEY in .streamlit/secrets.toml.")
    st.stop()

ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD")
MANAGER_PASSWORD = st.secrets.get("MANAGER_PASSWORD")
WORKER_PASSWORD = st.secrets.get("WORKER_PASSWORD")
if not all([ADMIN_PASSWORD, MANAGER_PASSWORD, WORKER_PASSWORD]):
    st.error("Role passwords not found in st.secrets. Please set ADMIN_PASSWORD, MANAGER_PASSWORD, and WORKER_PASSWORD in .streamlit/secrets.toml.")
    st.stop()

# ------------------------------------------------------------------------------
# Initialize models and vectorstore
# ------------------------------------------------------------------------------

# Set Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Configure OpenAI for YouTube transcription
openai_client = OpenAI(api_key=OPENAI_TRANSCRIPTION_API_KEY)

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
            # Load existing index
            vs = FAISS.load_local(
                FAISS_DIR, embedding_model, allow_dangerous_deserialization=True
            )
        else:
            # Create a new FAISS index with a compatible docstore
            from langchain_community.docstore.in_memory import InMemoryDocstore
            embedding_dim = len(embedding_model.embed_query("test"))
            index = faiss.IndexFlatL2(embedding_dim)
            vs = FAISS(
                embedding_function=embedding_model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            vs.save_local(FAISS_DIR)
        st.session_state.vectorstore = vs
    except Exception as e:
        st.error(f"Error initializing FAISS vectorstore: {e}")
        # Fallback to a new index
        from langchain_community.docstore.in_memory import InMemoryDocstore
        embedding_dim = len(embedding_model.embed_query("test"))
        index = faiss.IndexFlatL2(embedding_dim)
        vs = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vs.save_local(FAISS_DIR)
        st.session_state.vectorstore = vs

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def cleanup_temp_file(file_path: str):
    """Delete a temporary file."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        st.warning(f"Failed to delete temporary file {file_path}: {e}")

def clear_faiss_index():
    """Clear the FAISS index directory and reset the vectorstore."""
    try:
        if os.path.isdir(FAISS_DIR):
            shutil.rmtree(FAISS_DIR)
        from langchain_community.docstore.in_memory import InMemoryDocstore
        embedding_dim = len(embedding_model.embed_query("test"))
        index = faiss.IndexFlatL2(embedding_dim)
        vs = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vs.save_local(FAISS_DIR)
        st.session_state.vectorstore = vs
        st.session_state.uploaded_docs = []
        st.success("FAISS index and document list cleared successfully.")
    except Exception as e:
        st.error(f"Error clearing FAISS index: {e}")

def extract_text_from_pdf(file_path: str) -> str:
    """
    Try to extract text page by page.
    - Tries OCR loader first for all pages.
    - If that fails, tries PyPDFLoader page by page.
    - If a page fails, continues to the next one.
    No errors or warnings shown to user.
    """
    # Try OCR loader page by page
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(file_path, strategy="hi_res")
        pages = loader.load()
        texts = []
        for page in pages:
            try:
                texts.append(page.page_content)
            except Exception:
                continue  # Just skip bad pages
        if texts:
            return " ".join(texts)
    except Exception:
        pass  # Ignore all OCR errors

    # Fallback: PyPDFLoader, page by page
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path, extract_images=False)
        pages = loader.load()
        texts = []
        for page in pages:
            try:
                texts.append(page.page_content)
            except Exception:
                continue  # Just skip bad pages
        if texts:
            return " ".join(texts)
    except Exception:
        pass  # Ignore all errors

    # Nothing could be read, just return empty
    return ""


def extract_text_from_csv(file_path: str) -> str:
    """Convert CSV file to plain text by concatenating all fields."""
    try:
        df = pd.read_csv(file_path)
        text = " ".join(df.astype(str).apply(lambda row: " ".join(row.values), axis=1))
        if not text.strip():
            st.warning("No text could be extracted from the CSV.")
        return text
    except Exception as e:
        st.error(f"Error extracting text from CSV: {e}")
        return ""

def extract_text_from_excel(file_path: str) -> str:
    """Convert Excel file to plain text by concatenating all cells."""
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        text = " ".join(df.astype(str).apply(lambda row: " ".join(row.values), axis=1))
        if not text.strip():
            st.warning("No text could be extracted from the Excel file.")
        return text
    except Exception as e:
        st.error(f"Error extracting text from Excel: {e}")
        return ""

def extract_text_from_pptx(file_path: str) -> str:
    """Extract text from a PowerPoint file using UnstructuredPowerPointLoader."""
    try:
        loader = UnstructuredPowerPointLoader(file_path, mode="elements")
        pages = loader.load()
        text = " ".join(page.page_content for page in pages)
        if not text.strip():
            st.warning("No text could be extracted from the PowerPoint file.")
        return text
    except Exception as e:
        st.error(f"Error extracting text from PowerPoint: {e}")
        return ""

def extract_text_from_website(url: str) -> str:
    """Extract text from a website URL using AsyncHtmlLoader and Html2TextTransformer."""
    try:
        loader = AsyncHtmlLoader([url])
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        text = docs_transformed[0].page_content if docs_transformed else ""
        if not text.strip():
            st.warning(f"No text could be extracted from the website: {url}")
        return text
    except Exception as e:
        st.error(f"Error extracting text from website {url}: {e}")
        return ""

def extract_text_from_youtube(url: str) -> str:
    """Extract transcript from a YouTube video using yt-dlp and OpenAI Whisper."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': tmp_audio.name.replace(".mp3", ""),
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            with open(tmp_audio.name, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            text = transcript.text
            cleanup_temp_file(tmp_audio.name)
            if not text.strip():
                st.warning(f"No transcript could be extracted from the YouTube video: {url}")
            return text
    except Exception as e:
        st.error(f"Error extracting transcript from YouTube video {url}: {e}")
        return ""

def chunk_text(text: str, max_words: int = 500) -> list[tuple[int, str]]:
    """Break text into chunks of approximately max_words words."""
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

def index_document_in_vectorstore(doc_id: str, text: str):
    """Add document chunks to the FAISS vectorstore and persist."""
    try:
        chunks = chunk_text(text)
        documents = []
        for idx, chunk in chunks:
            metadata = {"doc_id": doc_id, "chunk_index": idx}
            documents.append(Document(page_content=chunk, metadata=metadata))
        if documents:
            st.session_state.vectorstore.add_documents(documents)
            st.session_state.vectorstore.save_local(FAISS_DIR)
    except Exception as e:
        st.error(f"Error indexing document into FAISS: {e}")

def index_file(doc_id: str, file_path: str, file_type: str, file_data: bytes):
    """Index content based on file type and return file data."""
    text = ""
    if file_type == "pdf":
        text = extract_text_from_pdf(file_path)
    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif file_type == "csv":
        text = extract_text_from_csv(file_path)
    elif file_type == "xlsx":
        text = extract_text_from_excel(file_path)
    elif file_type == "pptx":
        text = extract_text_from_pptx(file_path)
    if text.strip():
        index_document_in_vectorstore(doc_id, text)
    return file_data

def index_url(doc_id: str, url: str, input_type: str):
    """Index content from a website or YouTube URL with validation."""
    if input_type == "website_url":
        if not url.lower().startswith(("http://", "https://")):
            st.error(f"Invalid website URL: {url}. Please provide a valid URL starting with http:// or https://.")
            return
        text = extract_text_from_website(url)
    elif input_type == "youtube_url":
        if "youtube.com" not in url.lower() and "youtu.be" not in url.lower():
            st.error(f"Invalid YouTube URL: {url}. Please provide a valid YouTube URL (e.g., https://www.youtube.com/watch?v=...).")
            return
        text = extract_text_from_youtube(url)
    else:
        st.error(f"Invalid input type: {input_type}. Please select 'Website URL' or 'YouTube URL'.")
        return
    if text.strip():
        index_document_in_vectorstore(doc_id, text)
        return True
    return False

def generate_answer_with_gemini(question: str, context: str) -> str:
    """Generate an answer using Gemini."""
    try:
        input_prompt = prompt.format(context=context, question=question)
        response = gemini_model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating answer with Gemini: {e}")
        return "Unable to generate answer."

def search_vectorstore(question: str, top_k: int = 3) -> str:
    """
    Retrieve top_k relevant chunks from FAISS,
    but ONLY for docs the current user has access to.
    """
    try:
        # Only search for doc_ids that match accessible_roles
        current_role = st.session_state.get("user_role")
        uploaded_docs = st.session_state.get("uploaded_docs", [])
        allowed_roles = {"admin": ["admin", "manager", "worker"], "manager": ["manager", "worker"], "worker": ["worker"]}
        accessible_roles = allowed_roles.get(current_role, [])
        accessible_doc_ids = set(doc["doc_id"] for doc in uploaded_docs if doc["role"] in accessible_roles)
        # Get all matching chunks (FAISS is not role-aware)
        docs = st.session_state.vectorstore.similarity_search(question, k=top_k*5)  # Get more to allow stricter filter
        # Filter: only use chunks from allowed docs
        filtered_docs = [
            doc for doc in docs
            if doc.metadata.get("doc_id") in accessible_doc_ids
        ]
        if not filtered_docs:
            st.info("No relevant context found **for your access level**. Try uploading more documents or check your permissions.")
            return ""
        contexts = [doc.page_content for doc in filtered_docs[:top_k]]
        return "\n---\n".join(contexts)
    except Exception as e:
        st.error(f"Error searching vectorstore: {e}")
        return ""


# ------------------------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------------------------

# Apply custom styles
page_style()

st.title("RAG Knowledge System")

# Initialize session_state keys
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "login_status" not in st.session_state:
    st.session_state.login_status = False

# Role Selection and Login
def handle_login():
    role = st.session_state.role_select
    password = st.session_state.password_input
    password_map = {
        "admin": ADMIN_PASSWORD,
        "manager": MANAGER_PASSWORD,
        "worker": WORKER_PASSWORD
    }
    if password == password_map.get(role.lower()):
        st.session_state.user_role = role.lower()
        st.session_state.login_status = True
        st.success(f"Logged in as {role}.")
    else:
        st.error("Incorrect password. Please try again.")

if not st.session_state.login_status:
    with st.form(key="login_form"):
        st.subheader("Login")
        st.session_state.role_select = st.selectbox("Select Role:", ["Admin", "Manager", "Worker"])
        st.session_state.password_input = st.text_input("Enter Password:", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            handle_login()
else:
    st.write(f"Logged in as: {st.session_state.user_role.capitalize()}")
    if st.button("Logout"):
        st.session_state.user_role = None
        st.session_state.login_status = False
        st.success("Logged out successfully.")
        st.rerun()

# Show tabs only if logged in
if st.session_state.login_status:
    tab1, tab2, tab3 = st.tabs(["📤 Main (Upload)", "❓ Q&A", "📚 Document Library"])

    # ---------------------------- Main (Upload) -----------------------------------
    with tab1:
        st.header("Upload Documents or URLs")
        st.write("Upload .txt, .pdf, .csv, .xlsx, .pptx files, or input website/YouTube URLs, and assign role-based access.")

        if st.button("Clear Embeddings"):
            clear_faiss_index()

        input_type = st.selectbox("Input Type:", ["File", "Website URL", "YouTube URL"])
        assigned_role = st.selectbox("Assign access to role:", ["worker", "manager", "admin"])

        if input_type == "File":
            uploaded_file = st.file_uploader(
                "Select a file",
                type=["txt", "pdf", "csv", "xlsx", "pptx"],
                accept_multiple_files=False
            )
            if uploaded_file:
                st.markdown(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size // 1024} KB")
                # ADDED: file size check and error
                if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.error(f"File is too large! Please upload files smaller than {MAX_FILE_SIZE_MB}MB.")  # ADDED
                elif st.button("Upload and Index"):
                    file_data = uploaded_file.read()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                        tmp.write(file_data)
                        tmp.flush()
                        tmp_path = tmp.name

                    doc_id = f"file_{len(st.session_state.uploaded_docs) + 1}"
                    file_type = {
                        "application/pdf": "pdf",
                        "text/plain": "txt",
                        "text/csv": "csv",
                        "application/vnd.ms-excel": "xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
                    }.get(uploaded_file.type, "unknown")

                    if file_type != "unknown":
                        file_data = index_file(doc_id, tmp_path, file_type, file_data)
                        st.success(f"{file_type.upper()} file indexed into FAISS.")
                        st.session_state.uploaded_docs.append({
                            "doc_id": doc_id,
                            "filename": uploaded_file.name,
                            "path": tmp_path,
                            "role": assigned_role,
                            "type": file_type,
                            "file_data": file_data,
                        })
                        st.success(f"{file_type.upper()} '{uploaded_file.name}' uploaded for role '{assigned_role}'.")
                    else:
                        st.error("Unsupported file type.")
                        cleanup_temp_file(tmp_path)


        else:
            url_input = st.text_input(f"Enter {input_type}:")
            if url_input and st.button("Index URL"):
                doc_id = f"url_{len(st.session_state.uploaded_docs) + 1}"
                if index_url(doc_id, url_input, input_type.lower().replace(" ", "_")):
                    st.session_state.uploaded_docs.append({
                        "doc_id": doc_id,
                        "filename": url_input,
                        "path": None,
                        "role": assigned_role,
                        "type": input_type.lower().replace(" ", "_"),
                        "file_data": None,
                    })
                    st.success(f"{input_type} '{url_input}' indexed for role '{assigned_role}'.")

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
                else:
                    st.warning("No relevant context found. Try uploading more documents.")

    # ------------------------- Document Library ----------------------------------
    with tab3:
        st.header("Document Library")
        st.write("List of documents you have access to based on your role.")

        current_role = st.session_state.get("user_role")
        uploaded_docs = st.session_state.get("uploaded_docs", [])
        allowed_roles = {"admin": ["admin", "manager", "worker"], "manager": ["manager", "worker"], "worker": ["worker"]}
        accessible_roles = allowed_roles.get(current_role, [])
        docs = [doc for doc in uploaded_docs if doc["role"] in accessible_roles]

        if not docs:
            st.info("No documents available for your role.")
        else:
            table_data = []
            for doc in docs:
                type_label = (
                    doc["type"].upper() if doc["type"] in ["pdf", "txt", "csv", "xlsx", "pptx"]
                    else doc["type"].replace("_", " ").title()
                )
                table_data.append({
                    "Filename/URL": doc["filename"],
                    "Role": doc["role"],
                    "Type": type_label
                })
            df = pd.DataFrame(table_data)
            df.index += 1
            df.index.name = "No"
            st.table(df)

            for doc in docs:
                if doc.get("file_data"):
                    st.download_button(
                        label=f"Download {doc['filename']}",
                        data=doc["file_data"],
                        file_name=doc["filename"],
                        key=f"dl_{doc['doc_id']}_{doc['filename']}"  # <-- Unique key for each download button
                    )

#------------------ ADDED: Session/user warning -------------------------------
# ----------------- ADDED: About app error/limitation feedback ----------------
    st.caption("""
    ⚠️ This app is for single-session use only. Uploaded documents and answers are NOT shared across users or browser sessions.
            

    **Limitations:**
    - Large PDFs/videos may fail or be slow due to API quota or memory limits.
    - No multi-user or remote database sync: documents and answers are per-session only.
    - Error messages will appear below if anything fails (file, API, or parsing).
    """)