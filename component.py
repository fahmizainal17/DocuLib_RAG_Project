import streamlit as st
from PIL import Image
import base64

def get_base64_of_bin_file(bin_file):
    """
    Function to encode a local file (image or gif) to a base64 string.
    """
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def page_style():
    # Encode the local image to base64 for the sidebar background
    sidebar_image_base64 = get_base64_of_bin_file("assets/doc_background.jpg")

    # Custom CSS to style the page and sidebar
    custom_style = f"""
        <style>
            /* Hide Streamlit default elements */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}

            /* Sidebar background with a dark overlay */
            [data-testid="stSidebar"] > div:first-child {{
                background-color: #111;  /* Fallback solid dark background */
                background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)),
                url("data:image/jpg;base64,{sidebar_image_base64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: local;
            }}

            /* Adjust header and toolbar */
            [data-testid="stHeader"] {{
                background: rgba(0,0,0,0);
            }}

            [data-testid="stToolbar"] {{
                right: 2rem;
            }}

            /* Button styles */
            .stButton>button {{background-color: #FFA500; color: white !important;}}
            .stDownloadButton>button {{background-color: #FFA500; color: white !important;}}

            /* Custom card styles */
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

    # Set the page configuration with a custom icon and wide layout
    icon = Image.open("assets/RAG_LLM_Pic.jpg")
    st.set_page_config(page_title="DocuLib RAG System", page_icon=icon, layout="wide")

    # Apply the custom CSS to the page
    st.markdown(custom_style, unsafe_allow_html=True)

    # Display the main background image at the top of the page
    main_bg = Image.open("assets/DocuLib_Background.png")
    st.image(main_bg, use_container_width=True)

    # Sidebar content
    with st.sidebar:
        # Display a round profile picture (or logo) at the top of the sidebar
        st.image("photos/Round_Profile_Photo.png", width=100)

        # --- DocuLib RAG System Sidebar Content ---
        st.markdown("""
            ## 📂 DocuLib RAG System

            **Learn and retrieve knowledge across all your documents through levels of access.**

            ### 📖 How to Use:
            1. Upload any `.txt`, `.pdf`, `.csv`, or `.xlsx` file in the “Upload” tab.  
            2. Assign a role (worker, manager, admin) so only those roles can access each document.  
            3. In the “Q&A” tab, ask questions and get instant answers based on your uploaded content.  
            4. Use the “Document Library” tab to view or download files you have permission to access.

            ### 🤖 Technologies Used:
            - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`  
            - **Vector Store:** FAISS (via LangChain)  
            - **Text Generation:** Gemini 2.0 Flash (Google Generative AI)  
            - **Database & Auth:** Supabase

            ### 🔑 User Roles:
            - **Worker:** Can upload and query documents tagged “worker”.  
            - **Manager:** Can upload and query documents tagged “manager” or “worker”.  
            - **Admin:** Full access to all documents.

            ---
        """)

        # Optional: Play background music button
        new_tab_button = """
        <a href="https://youtu.be/kx5N2TeDqNM?si=-sCwGJpuKLQ1PFO6" target="_blank">
            <button style="background-color: #FFA500; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
                🎵 Play Background Music
            </button>
        </a>
        """
        st.markdown(new_tab_button, unsafe_allow_html=True)

        st.markdown("""---""")

        # About the Developer or Team
        st.markdown("""
        ### 👨‍💻 About the Developer
        Hi! We are the **DocuLib Team**, dedicated to making document retrieval and knowledge management simple and secure.

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
