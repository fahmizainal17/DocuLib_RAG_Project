# **📂 DocuLib RAG Knowledge System**

<div align="center">
  <a href="https://doculib-rag-app.streamlit.app/">
    <img
      src="https://img.shields.io/badge/Streamlit-DocuLib_RAG_System-blue?style=for-the-badge&logo=streamlit"
      alt="DocuLib RAG System"
    />
  </a>
</div>

## ![DocuLib RAG System Screenshot](assets/app_image.png)

---

## **📄 Overview**

The **DocuLib RAG Knowledge System** is a Streamlit-based, role-aware Retrieval-Augmented Generation (RAG) application.  
It enables secure document upload, semantic search, and question-answering over multiple file types and URLs.  
Users (Admin, Manager, Worker) authenticate with passwords, upload documents or URLs, and query content via a FAISS vector store.  
Answers are generated using Google’s Gemini 2.0 Flash model.

---

## **🛠️ System Architecture**

**High-Level System Flow:**
```mermaid
flowchart LR
    User([User])
    StreamlitApp["Streamlit App<br>(DocuLib RAG System)"]
    Auth[(st.secrets Auth)]
    FAISS[(FAISS Index)]
    Gemini[(Gemini 2.0 Flash)]
    Storage[(Local Temp Files)]
    Session[(Session State)]

    User --> StreamlitApp
    StreamlitApp --> Auth
    StreamlitApp --> FAISS
    StreamlitApp --> Gemini
    StreamlitApp --> Storage
    StreamlitApp --> Session
    FAISS <--> Storage
    Gemini <--> FAISS

    style StreamlitApp fill:#FF4B4B,stroke:#C62E2E,stroke-width:2px
    style FAISS fill:#4C6EF5,stroke:#364FC7,stroke-width:2px
    style Gemini fill:#B197FC,stroke:#7048E8,stroke-width:2px
    style Storage fill:#A5D8FF,stroke:#339AF0,stroke-width:2px
    style Auth fill:#C5F6FA,stroke:#22B8CF,stroke-width:2px
    style Session fill:#E9FAC8,stroke:#90C695,stroke-width:2px
````

### **Backend**

* **Application Layer:**

  * Python, **Streamlit** for fast web UI.
* **Authentication:**

  * Role-based, password-protected using secrets in `.streamlit/secrets.toml`.
* **Embedding & Search:**

  * **LangChain** + **FAISS** for semantic search/retrieval.
  * Embeddings via `sentence-transformers/all-MiniLM-L6-v2`.
* **Answer Generation:**

  * **Google Gemini 2.0 Flash** for Q\&A.
* **File Extraction:**

  * `.txt`, `.pdf`, `.csv`, `.xlsx`, `.pptx` via loaders.
  * URLs (websites, YouTube) via AsyncHtmlLoader and yt-dlp + Whisper.

### **Frontend**

* **User Interface:**

  * Modern, responsive UI with custom CSS (see `component/page_style.py`).
  * Tabs for Main (Upload), Q\&A, Document Library.
  * Real-time, multi-role, document management and downloads.

### **Data Storage**

* **Document Embeddings:**

  * Locally in `faiss_index/`.
* **User Session & Uploaded Files:**

  * Managed in `st.session_state`.
* **Temp Files:**

  * Secure, cleaned up after use.

---

## **🎯 Key Features**

* **🔐 Role-Based Access Control**

  * Three user roles (Admin, Manager, Worker) with password authentication.
  * Granular document upload and Q\&A access per role.

* **📤 Flexible Document & URL Ingestion**

  * Supports `.txt`, `.pdf`, `.csv`, `.xlsx`, `.pptx`, website URLs, and YouTube video transcriptions.

* **🤖 Semantic Search & Answer Generation**

  * Top-3 relevant chunks retrieved from FAISS.
  * Answers (50–75 words) using Google Gemini 2.0 Flash.

* **📚 Document Library & Secure Downloads**

  * Role-filtered table of uploaded documents, secure file downloads.

* **🎨 Custom UI**

  * Streamlit-based with custom backgrounds, sidebar, and wide layout.

---

## **⚠️ Known Limitations**

* **Single User Session:**

  * No persistent user accounts or audit logs; authentication is session-based.
* **Local Storage:**

  * All data, vector index, and uploaded files are local; **no cloud database**.
* **File Size & API Quotas:**

  * Large PDFs and long videos may be slow to process; subject to API limits and file size.
* **No Multi-user Sync:**

  * Not designed for concurrent editing or remote database integration out of the box.
* **Limited Error Handling:**

  * Some edge cases (corrupt files, OCR failures, API downtime) may require manual intervention.
* **Gemini/OpenAI Dependency:**

  * Relies on external APIs; service downtime or quota exhaustion affects Q\&A and transcription.

---

## **Table of Contents**

1. [🎯 Key Features](#-key-features)
2. [🛠️ System Architecture](#️-system-architecture)
3. [🔧 Technology Stack](#-technology-stack)
4. [📝 Project Structure](#-project-structure)
5. [🚀 Getting Started](#-getting-started)
6. [🔒 Authentication & Roles](#-authentication--roles)
7. [📤 Main (Upload) Tab](#-main-upload-tab)
8. [❓ Q\&A Tab](#-qna-tab)
9. [📚 Document Library Tab](#-document-library-tab)
10. [📂 Underlying Components](#-underlying-components)
11. [🛠️ Development & Testing](#-development--testing)
12. [📚 References](#-references)
13. [📜 License](#-license)

---

## **🔧 Technology Stack**

![Python](https://img.shields.io/badge/python-3.12+-3670A0?style=for-the-badge\&logo=python\&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.0+-000000?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-1.7.4-blue?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-purple?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-0.27+-412991?style=for-the-badge\&logo=openai\&logoColor=white)
![yt-dlp](https://img.shields.io/badge/yt--dlp-2025.05-green?style=for-the-badge)
![faiss](https://img.shields.io/badge/faiss-1.7.4-blue?style=for-the-badge)

---

## **📝 Project Structure**

```plaintext
.
├── assets/
│   ├── DocuLib_Background.png   # Main page background
│   ├── RAG_LLM_Pic.jpg          # Favicon / app icon
│   └── doc_background.jpg       # Sidebar background image
├── images/
│   └── app_image.png            # README screenshot
├── main.py                      # Streamlit application entry point
├── component/
│   └── page_style.py            # Custom CSS and page configuration
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation (this file)
└── .streamlit/
    └── secrets.toml             # Passwords & API keys (not in repo)
```

---

## **🚀 Getting Started**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/fahmizainal17/DocuLib_RAG_System.git
   cd DocuLib_RAG_System
   ```

2. **Create & Activate Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Secrets**

   * In `.streamlit/secrets.toml`, add:

     ```toml
     HF_TOKEN = "<YOUR_HUGGINGFACE_TOKEN>"
     GEMINI_API_KEY = "<YOUR_GEMINI_API_KEY>"
     OPENAI_TRANSCRIPTION_API_KEY = "<YOUR_OPENAI_WHISPER_KEY>"
     ADMIN_PASSWORD = "<ADMIN_PASSWORD>"
     MANAGER_PASSWORD = "<MANAGER_PASSWORD>"
     WORKER_PASSWORD = "<WORKER_PASSWORD>"
     ```
   * **Important**: Do not commit `secrets.toml` to source control.

5. **Run the App**

   ```bash
   streamlit run main.py
   ```

6. **Access in Browser**
   Navigate to `http://localhost:8501`. Log in using one of the three roles and corresponding password.

---

## **🔒 Authentication & Roles**

Upon launch, users must select a role and enter the matching password:

* **Admin:**
  Full access to all documents (admin, manager, worker).
* **Manager:**
  Access to documents tagged “manager” and “worker”.
* **Worker:**
  Access only to documents tagged “worker”.

---

## **📤 Main (Upload) Tab**

* Upload `.txt`, `.pdf`, `.csv`, `.xlsx`, `.pptx` files, or input website/YouTube URLs.
* Assign the uploaded file to a user role.
* Extraction and chunking handled automatically for each file type or URL.

---

## **❓ Q\&A Tab**

* Enter a question and click **Get Answer**.
* The system searches for relevant content (filtered by your role) and generates a concise answer using Gemini.

---

## **📚 Document Library Tab**

* Displays all documents your role can access.
* Secure file downloads for uploaded documents.

---

## **📂 Underlying Components**

* See source code and docstrings in `main.py` for details on:

  * FAISS initialization and saving
  * Chunking logic
  * File extraction methods
  * Vectorstore indexing

---

## **🛠️ Development & Testing**

* Install dev dependencies:
  `pip install -r requirements.txt`
* Local linter:
  `flake8 .`
* (Optional) Unit tests:
  `pytest tests/ -v`
* Streamlit debug mode:
  `streamlit run main.py --server.runOnSave true`
* All temp files are auto-cleaned.

---

## **📚 References**

* [Streamlit Documentation](https://docs.streamlit.io/)
* [LangChain FAISS Integration](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/faiss.html)
* [Google Gemini API (Flash)](https://developers.generativeai.google)
* [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
* [yt-dlp Repository](https://github.com/yt-dlp/yt-dlp)
* [PyPDFLoader (LangChain)](https://python.langchain.com/en/latest/modules/document_loaders/external/pdf.html)
* [AsyncHtmlLoader & Html2TextTransformer](https://python.langchain.com/en/latest/modules/document_loaders/external/html.html)

---

## **📜 License**

Fahmi Zainal Custom License
Unauthorized copying, distribution, or modification of this project is prohibited.
For licensing inquiries, contact the project maintainers.

---

*Last updated: June 2025*

