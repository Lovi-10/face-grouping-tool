import streamlit as st
import os
from PIL import Image
import base64
import urllib.parse
from face_grouper.gdrive_utils import download_gdrive_folder
from face_grouper.main import run_pipeline

# Directories
DOWNLOAD_DIR = "downloaded_photos"
OUTPUT_DIR = "output_faces"

st.set_page_config(layout="wide")
st.title("🧠 Face Grouping App")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "main"
if "selected_group" not in st.session_state:
    st.session_state.selected_group = None

# Handle direct URL query param
query_params = st.query_params
if "group" in query_params:
    st.session_state.page = "details"
    st.session_state.selected_group = query_params["group"]

# Util: get base64 thumbnail
def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode("utf-8")

# --- Page 1: Main UI ---
if st.session_state.page == "main":
    source_type = st.radio("Choose source type", ["Google Drive Folder URL", "Local Directory"])

    if source_type == "Google Drive Folder URL":
        gdrive_url = st.text_input("Enter Google Drive Folder URL")
        if st.button("Start Grouping"):
            with st.spinner("Downloading and processing..."):
                download_gdrive_folder(gdrive_url, DOWNLOAD_DIR)
                run_pipeline(DOWNLOAD_DIR, OUTPUT_DIR)
            st.success("✅ Done!")

    elif source_type == "Local Directory":
        local_path = st.text_input("Enter local image folder path")
        if st.button("Start Grouping"):
            with st.spinner("Processing local folder..."):
                run_pipeline(local_path, OUTPUT_DIR)
            st.success("✅ Done!")

    # Show face groups
    if os.path.exists(OUTPUT_DIR):
        folders = sorted(
            [f for f in os.listdir(OUTPUT_DIR) if f.startswith("person_")],
            key=lambda x: -len(os.listdir(os.path.join(OUTPUT_DIR, x)))
        )

        st.subheader("👥 Grouped Faces")
        cols_per_row = 5
        for i in range(0, len(folders), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, folder in enumerate(folders[i:i + cols_per_row]):
                group_path = os.path.join(OUTPUT_DIR, folder)
                thumb_path = os.path.join(group_path, "thumbnail.jpg")
                if os.path.exists(thumb_path):
                    b64_thumb = get_base64_image(thumb_path)
                    with cols[j]:
                        folder_encoded = urllib.parse.quote(folder)
                        st.markdown(
                            f"""
                            <a href="?group={folder_encoded}">
                                <img src="data:image/jpeg;base64,{b64_thumb}" 
                                     style="width:100px; height:100px; border-radius:50%; object-fit:cover; border:2px solid #ccc; display: block; margin: auto;" />
                            </a>
                            """,
                            unsafe_allow_html=True
                        )
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
# --- Page 2: Group Details ---
elif st.session_state.page == "details":
    group = st.session_state.selected_group
    group_path = os.path.join(OUTPUT_DIR, group)

    st.markdown("### 👤 Photos in this Group")
    if st.button("🔙 Back to Groups"):
        st.session_state.page = "main"
        st.session_state.selected_group = None
        st.query_params.clear()

    images = [
        os.path.join(group_path, f)
        for f in os.listdir(group_path)
        if f != "thumbnail.jpg"
    ]

    cols_per_row = 4
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, img_path in enumerate(images[i:i + cols_per_row]):
            with cols[j]:
                st.image(Image.open(img_path), use_container_width=True)