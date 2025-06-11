# DocuLib RAG System - Component for Streamlit App
import streamlit as st
from PIL import Image
import base64
import os

def get_base64_of_bin_file(bin_file):
    """Encode a local file (image or gif) to a base64 string."""
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def page_style():
    # Path safety: allow both 'assets/' and './assets/'
    sidebar_image_path = "assets/doc_background.jpg"
    if not os.path.exists(sidebar_image_path):
        sidebar_image_path = "./assets/doc_background.jpg"
    sidebar_image_base64 = get_base64_of_bin_file(sidebar_image_path)

    custom_style = f"""
        <style>
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}

            [data-testid="stSidebar"] > div:first-child {{
                background-color: #111;
                background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)),
                url("data:image/jpg;base64,{sidebar_image_base64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: local;
            }}

            [data-testid="stHeader"] {{
                background: rgba(0,0,0,0);
            }}

            [data-testid="stToolbar"] {{
                right: 2rem;
            }}

            .stButton>button {{background-color: #FFA500; color: white !important;}}
            .stDownloadButton>button {{background-color: #FFA500; color: white !important;}}

            .cert-card {{
                background-color: #333333;
                color: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .cert-card:hover {{
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }}
        </style>
    """

    # Set page config only once
    icon_path = "assets/RAG_LLM_Pic.jpg"
    if not os.path.exists(icon_path):
        icon_path = "./assets/RAG_LLM_Pic.jpg"
    icon = Image.open(icon_path)
    st.set_page_config(page_title="DocuLib RAG System", page_icon=icon, layout="wide")

    st.markdown(custom_style, unsafe_allow_html=True)

    main_bg_path = "assets/DocuLib_Background.png"
    if not os.path.exists(main_bg_path):
        main_bg_path = "./assets/DocuLib_Background.png"
    main_bg = Image.open(main_bg_path)
    st.image(main_bg, use_container_width=True)

    with st.sidebar:
        profile_photo_path = "photos/Round_Profile_Photo.png"
        if not os.path.exists(profile_photo_path):
            profile_photo_path = "./photos/Round_Profile_Photo.png"
        st.image(profile_photo_path, width=100)

        st.markdown("""
            ## 📂 DocuLib RAG System

            **Securely upload, manage, and query documents with role-based access.**

            ### 🔐 Role-Based Login
            1. Choose a role (Admin, Manager, Worker) and enter the corresponding password.  
               - **Admin:** Access all files (admin, manager, worker).  
               - **Manager:** Access manager and worker files only.  
               - **Worker:** Access worker files only.  
            2. Passwords are stored securely in `st.secrets` (ADMIN_PASSWORD, MANAGER_PASSWORD, WORKER_PASSWORD).

            ### 📤 Main (Upload) Tab
            1. Select “File”, “Website URL”, or “YouTube URL”.  
            2. Upload `.txt`, `.pdf`, `.csv`, `.xlsx`, or `.pptx` files **or** provide a valid URL.  
            3. Assign to a role.  
            4. Click “Upload and Index” (or “Index URL”).

            ### ❓ Q&A Tab
            - Enter a question and get answers from documents your role can access.

            ### 📚 Document Library Tab
            - View and download all documents your role can access.

            ### 🤖 Technologies Used
            - `PyPDFLoader`, `AsyncHtmlLoader`, `UnstructuredPowerPointLoader`, `yt-dlp` + OpenAI Whisper, Pandas
            - `sentence-transformers/all-MiniLM-L6-v2`, FAISS, Gemini 2.0 Flash, Streamlit

            ### 🔑 User Roles at a Glance
            - **Worker:**  
              • Upload & query documents tagged “worker”.
            - **Manager:**  
              • Upload & query documents tagged “manager” or “worker”.
            - **Admin:**  
              • Full access to documents tagged “admin”, “manager”, or “worker”.
            ---
        """)

        # Play background music (optional)
        st.markdown("""
        <a href="https://youtu.be/kx5N2TeDqNM?si=-sCwGJpuKLQ1PFO6" target="_blank">
            <button style="background-color: #FFA500; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
                🎵 Play Background Music
            </button>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("""---""")

        st.markdown("""
        ### 👨‍💻 About the Developer
        We are the **DocuLib Team**, dedicated to secure document retrieval and knowledge management.

        **Connect with us:**
        """)

        # LinkedIn Button
        linkedin_url = "https://www.linkedin.com/in/fahmizainal17"
        st.markdown(f"""
            <a href="{linkedin_url}" target="_blank">
                <button style="background-color: #0077B5; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
                    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" style="vertical-align: middle;"> Connect on LinkedIn
                </button>
            </a>
        """, unsafe_allow_html=True)

        # GitHub Button
        github_url = "https://github.com/fahmizainal17/DocuLib"
        st.markdown(f"""
            <a href="{github_url}" target="_blank">
                <button style="background-color: #333; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
                    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="16" style="vertical-align: middle;"> Check out DocuLib on GitHub
                </button>
            </a>
        """, unsafe_allow_html=True)
