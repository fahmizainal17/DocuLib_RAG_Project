"""
Streamlit RAG Knowledge System with Role-Based Access and st.secrets

This Streamlit app provides:
1. Role-based access control: documents can be tagged as "admin", "manager", or "worker".
2. A "Main" tab to upload .txt, .pdf, .csv, .xlsx, .pptx files, or input website URLs or YouTube links, which are immediately indexed into FAISS via LangChain.
   - PDF text is extracted by PyPDFLoader and manually chunked (~500 words).
   - Website content is extracted using AsyncHtmlLoader and Html2TextTransformer.
   - YouTube videos are transcribed using yt-dlp and OpenAI Whisper.
   - PowerPoint files are extracted using UnstructuredPowerPointLoader.
3. A "Q&A" tab to ask questions; the app retrieves relevant document chunks via FAISS and generates an answer using Gemini (google.generativeai).
4. A "Document Library" tab listing all uploaded documents filtered by the user’s role, displayed in a table with file type.
5. All secrets (HF token, Gemini API key, OpenAI API key) come from st.secrets.
"""

import os
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
from component import page_style  # Your custom CSS + sidebar styling

# Apply nest_asyncio for AsyncHtmlLoader compatibility
nest_asyncio.apply()

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
    st.error("OpenAI API key not found in st.secrets. Please set OPENAI_TRANSCRIPTION_API_KEY in .streamlit/secrets.toml for YouTube transcription.")
    st.stop()

# ------------------------------------------------------------------------------
# Initialize models and vectorstore
# ------------------------------------------------------------------------------

# Set Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")  # Using requested model

# Configure OpenAI for YouTube transcription
openai_client = OpenAI(api_key=OPENAI_TRANSCRIPTION_API_KEY)

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
        os.unlink(file_path)
    except Exception as e:
        st.warning(f"Failed to delete temporary file {file_path}: {e}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyPDFLoader."""
    try:
        loader = PyPDFLoader(file_path, extract_images=True)
        pages = loader.load()
        text = " ".join(page.page_content for page in pages)
        if not text.strip():
            st.warning("No text could be extracted from the PDF.")
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
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
        # Download audio using yt-dlp
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
            # Transcribe audio using OpenAI Whisper
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

def index_document_in_vectorstore(doc_id: int, text: str):
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

def index_file(doc_id: int, file_path: str, file_type: str):
    """Index content based on file type."""
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

def index_url(doc_id: int, url: str, input_type: str):
    """Index content from a website or YouTube URL."""
    text = extract_text_from_website(url) if input_type == "website" else extract_text_from_youtube(url)
    if text.strip():
        index_document_in_vectorstore(doc_id, text)

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
    """Retrieve top_k relevant chunks from FAISS, filtered by user role."""
    try:
        docs = st.session_state.vectorstore.similarity_search(question, k=top_k)
        current_role = st.session_state["user_role"]
        filtered_docs = [
            doc for doc in docs
            if any(
                d["doc_id"] == doc.metadata["doc_id"] and d["role"] == current_role
                for d in st.session_state["uploaded_docs"]
            )
        ]
        contexts = [doc.page_content for doc in filtered_docs]
        return "\n---\n".join(contexts) if contexts else ""
    except Exception as e:
        st.error(f"Error searching vectorstore: {e}")
        return ""

# ------------------------------------------------------------------------------
# Streamlit App: Role Selection & Tabs
# ------------------------------------------------------------------------------

# Apply custom styles
page_style()

st.title("RAG Knowledge System")

# Initialize session_state keys
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None
if "uploaded_docs" not in st.session_state:
    # Each entry: {"doc_id": int, "filename": str, "path": str, "role": str, "type": str}
    st.session_state["uploaded_docs"] = []

# Role selection
st.sidebar.header("Select Your Role")
role_choice = st.sidebar.selectbox("Role:", ["worker", "manager", "admin"])
st.session_state["user_role"] = role_choice

# Main Tabs
tab1, tab2, tab3 = st.tabs(
    ["📤 Main (Upload)", "❓ Q&A", "📚 Document Library"]
)

# ---------------------------- Main (Upload) -----------------------------------
with tab1:
    st.header("Upload Documents or URLs")
    st.write("Upload .txt, .pdf, .csv, .xlsx, .pptx files, or input website/YouTube URLs, and assign role-based access.")

    input_type = st.selectbox("Input Type:", ["File", "Website URL", "YouTube URL"])
    assigned_role = st.selectbox("Assign access to role", ["worker", "manager", "admin"])

    if input_type == "File":
        uploaded_file = st.file_uploader(
            "Select a file",
            type=["txt", "pdf", "csv", "xlsx", "pptx"],
            accept_multiple_files=False,
        )
        if uploaded_file:
            st.markdown(
                f"**File:** {uploaded_file.name}  |  **Size:** {uploaded_file.size // 1024} KB"
            )
            if st.button("Upload and Index"):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f"_{uploaded_file.name}"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp.flush()
                    tmp_path = tmp.name

                doc_id = len(st.session_state["uploaded_docs"]) + 1
                file_type = {
                    "application/pdf": "pdf",
                    "text/plain": "txt",
                    "text/csv": "csv",
                    "application/vnd.ms-excel": "xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
                }.get(uploaded_file.type, "unknown")

                if file_type != "unknown":
                    index_file(doc_id, tmp_path, file_type)
                    st.success(f"{file_type.upper()} file indexed into FAISS.")
                    st.session_state["uploaded_docs"].append(
                        {
                            "doc_id": doc_id,
                            "filename": uploaded_file.name,
                            "path": tmp_path,
                            "role": assigned_role,
                            "type": file_type,
                        }
                    )
                    st.success(
                        f"{file_type.upper()} '{uploaded_file.name}' uploaded for role '{assigned_role}'."
                    )
                    cleanup_temp_file(tmp_path)
                else:
                    st.error("Unsupported file type.")
                    cleanup_temp_file(tmp_path)

    else:  # Website or YouTube URL
        url_input = st.text_input(f"Enter {input_type}:")
        if url_input and st.button("Index URL"):
            doc_id = len(st.session_state["uploaded_docs"]) + 1
            index_url(doc_id, url_input, input_type.lower().replace(" ", "_"))
            st.success(f"{input_type} indexed into FAISS.")
            st.session_state["uploaded_docs"].append(
                {
                    "doc_id": doc_id,
                    "filename": url_input,
                    "path": None,  # No local file for URLs
                    "role": assigned_role,
                    "type": input_type.lower().replace(" ", "_"),
                }
            )
            st.success(
                f"{input_type} '{url_input}' indexed for role '{assigned_role}'."
            )

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
        doc for doc in st.session_state["uploaded_docs"]
        if doc["role"] == current_role
    ]

    if not docs:
        st.info("No documents available for your role.")
    else:
        # Display documents in a table
        table_data = [
            {
                "Filename/URL": doc["filename"],
                "Role": doc["role"],
                "Type": doc["type"].upper() if doc["type"] in ["pdf", "txt", "csv", "xlsx", "pptx"] else doc["type"].replace("_", " ").title()
            }
            for doc in docs
        ]
        st.table(table_data)

        # Display download buttons for file-based documents
        for doc in docs:
            if doc["path"]:  # Only files have a path
                st.markdown(f"**{doc['filename']}** (Role: {doc['role']}, Type: {doc['type'].upper()})")
                with open(doc["path"], "rb") as f:
                    file_data = f.read()
                st.download_button(
                    label=f"Download {doc['filename']}",
                    data=file_data,
                    file_name=doc["filename"],
                )
            else:  # URLs
                st.markdown(f"**{doc['filename']}** (Role: {doc['role']}, Type: {doc['type'].replace('_', ' ').title()})")