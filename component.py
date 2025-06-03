import streamlit as st
from PIL import Image
import base64

def get_base64_of_bin_file(bin_file):
    """
    Function to encode local file (image or gif) to base64 string
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def page_style():
    # Encode the local image to base64
    sidebar_image_base64 = get_base64_of_bin_file('assets/doc_background.jpg')

    # Apply custom styles, including the sidebar background image
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

    # Set the page configuration with a custom icon
    icon = Image.open('assets/RAG_LLM_Pic.jpg')
    st.set_page_config(page_title="Fahmi's Resume Q&A", page_icon=icon, layout="wide")

    # Apply custom styles to the page
    st.markdown(custom_style, unsafe_allow_html=True)

    # Display the main background image
    image = Image.open('assets/DocuLib_Background.png')
    st.image(image)

    # Sidebar content
    with st.sidebar:
        # Display the round profile picture at the top of the sidebar
        st.image("photos/Round_Profile_Photo.png", width=100)

        # --- New RAG Project Sidebar Content ---
        st.markdown("""
            ## 📄 Fahmi's Resume Q&A (RAG Project)

            **Explore Muhammad Fahmi's professional journey through this intelligent Q&A app!**

            ### 🔍 How to Use:
            1. Enter any question about Fahmi's experience or skills.  
            2. Get an answer instantly based on his resume.

            ### 🤖 Models Used:
            - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` for semantic search.
            - **Text Generation Model:** `Gemini 1.5 Flash` for generating concise answers.

            ### 🗂️ Data Stored:
            - Fahmi's resume is stored in a **FAISS Vector Database** for fast retrieval.

            ### 💡 Example Questions:
            - *"What are Fahmi's technical skills?"*  
            - *"Tell me about Fahmi's role at Invoke Solutions."*

            ---
        """)

        # Play Background Music Button
        new_tab_button = """
        <a href="https://youtu.be/kx5N2TeDqNM?si=-sCwGJpuKLQ1PFO6" target="_blank">
            <button style="background-color: #FFA500; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
                🎵 Play Background Music
            </button>
        </a>
        """
        st.markdown(new_tab_button, unsafe_allow_html=True)

        st.markdown("""---""")

        # About the Developer
        st.markdown("""
        ### 👨‍💻 About the Developer
        Hi! I'm **Fahmi Zainal**, a data scientist and developer passionate about building interactive applications using AI and Machine Learning.

        **Connect with me:**
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
        github_url = "https://github.com/fahmizainal17"
        st.markdown(f"""
            <a href="{github_url}" target="_blank">
                <button style="background-color: #333; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
                    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="16" style="vertical-align: middle;"> Check out my GitHub
                </button>
            </a>
        """, unsafe_allow_html=True)
