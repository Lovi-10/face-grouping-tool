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
st.markdown("""
<div style="text-align:center; margin-top: -30px;">
    <h1 style="font-size: 2.5em;">🧠 Face Grouping Tool</h1>
    <p style="color:gray;">Upload or fetch images, detect faces, and group them by similarity.</p>
</div>
""", unsafe_allow_html=True)


# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "main"
if "selected_group" not in st.session_state:
    st.session_state.selected_group = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "source_type" not in st.session_state:
    st.session_state.source_type = "Google Drive Folder URL"

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
    st.markdown("### 📂 Select Image Source")
    source_type = st.radio(
        "",
        ["Google Drive Folder URL", "Local Directory"],
        horizontal=True,
        key="source_type",
        disabled=st.session_state.processing
    )

    if st.session_state.source_type == "Google Drive Folder URL":
        with st.container():
            st.markdown("#### 🔗 Fetch Images from Google Drive")
            gdrive_url = st.text_input("Paste Google Drive Folder URL here")
            st.caption("Make sure the folder is shared publicly")

            if st.button("🚀 Start Grouping") and not st.session_state.processing:
                st.session_state.processing = True
                with st.spinner("Downloading and processing..."):

                    progress = st.progress(0, text="Downloading images...")


                    def update_progress(fraction):
                        progress.progress(fraction, text=f"Downloading {int(fraction * 100)}%")


                    download_gdrive_folder(gdrive_url, DOWNLOAD_DIR, progress_callback=update_progress)
                    progress.empty()

                    progress_bar = st.progress(0, text="Processing images...")


                    def update_processing(fraction):
                        progress_bar.progress(fraction, text=f"Processing {int(fraction * 100)}%")


                    run_pipeline(DOWNLOAD_DIR, OUTPUT_DIR, update_progress=update_processing)
                    progress_bar.empty()
                st.session_state.processing = False
                st.success("✅ Done!")


    elif st.session_state.source_type == "Local Directory":
        with st.container():
            st.markdown("#### 📁 Upload Images from Local Folder")
            uploaded_files = st.file_uploader(
                "Choose image files",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            st.caption("Supported formats: JPG, JPEG, PNG")

            if st.button("🚀 Start Grouping") and not st.session_state.processing:
                st.session_state.processing = True
                os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                progress = st.progress(0, text="Saving uploaded images...")
                total_files = len(uploaded_files)

                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    progress.progress((i + 1) / total_files, text=f"Saved {i + 1}/{total_files} images")

                progress.empty()
                progress_bar = st.progress(0, text="Processing images...")


                def update_progress(fraction):
                    progress_bar.progress(fraction, text=f"Processing {int(fraction * 100)}%")


                run_pipeline(DOWNLOAD_DIR, OUTPUT_DIR, update_progress=update_progress)
                progress_bar.empty()
                st.session_state.processing = False
                st.success("✅ Done!")

    # Show face groups
    if os.path.exists(OUTPUT_DIR):
        folders = sorted(
            [f for f in os.listdir(OUTPUT_DIR) if f.startswith("person_")],
            key=lambda x: -len(os.listdir(os.path.join(OUTPUT_DIR, x)))
        )

        st.markdown("---")
        st.subheader("👥 Grouped Faces")
        st.caption("Click a face to view all photos of that person")

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
                                     style="width:100px; height:100px; border-radius:50%; object-fit:cover; border:2px solid #ccc; background-color:#f9f9f9; padding:4px; display: block; margin: auto;" />
                            </a>
                            """,
                            unsafe_allow_html=True
                        )
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
# --- Page 2: Group Details ---
elif st.session_state.page == "details":
    group = st.session_state.selected_group
    group_path = os.path.join(OUTPUT_DIR, group)

    st.markdown(f"<h3 style='text-align:center;'>👤 Photos in <code>{group}</code></h3>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
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