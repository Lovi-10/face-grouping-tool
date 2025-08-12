import streamlit as st
import os
from PIL import Image
import base64
from face_grouper.gdrive_utils import download_gdrive_folder
from face_grouper.main import run_pipeline

# Directories
DOWNLOAD_DIR = "downloaded_photos"
OUTPUT_DIR = "output_faces"

st.set_page_config(
    page_title="Face Grouping Tool", 
    page_icon="ðŸ‘ï¸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with enhanced hover effects
def inject_modern_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {background-color: transparent;}
    
    /* Main container */
    .main-content {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 2rem auto;
        max-width: 1400px;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Header */
    .hero-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #64748b;
        font-weight: 400;
    }
    
    /* Enhanced Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 16px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        color: white;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 16px 40px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(1.01);
    }
    
    /* Thumbnail Grid Container */
    .thumbnail-section {
        margin-top: 3rem;
        padding: 2rem 0;
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .section-subtitle {
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Enhanced Thumbnail Cards */
    .thumbnail-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 2rem;
        padding: 1rem;
    }
    
    .thumbnail-wrapper {
        position: relative;
        cursor: pointer;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .thumbnail-card {
        background: white;
        border-radius: 20px;
        padding: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 2px solid transparent;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .thumbnail-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: 20px;
        z-index: 1;
    }
    
    .thumbnail-wrapper:hover {
        transform: translateY(-12px) scale(1.03);
        z-index: 10;
    }
    
    .thumbnail-wrapper:hover .thumbnail-card {
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .thumbnail-wrapper:hover .thumbnail-card::before {
        opacity: 1;
    }
    
    .thumbnail-image {
        width: 100%;
        height: 140px;
        object-fit: cover;
        border-radius: 16px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        z-index: 2;
    }
    
    .thumbnail-wrapper:hover .thumbnail-image {
        transform: scale(1.05);
        filter: brightness(1.1) contrast(1.05);
    }
    
    .thumbnail-info {
        padding: 1rem 0.5rem 0.5rem 0.5rem;
        text-align: center;
        position: relative;
        z-index: 2;
    }
    
    .thumbnail-label {
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
    }
    
    .thumbnail-count {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .thumbnail-badge {
        position: absolute;
        top: 16px;
        right: 16px;
        background: rgba(0, 0, 0, 0.75);
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
        z-index: 3;
        transition: all 0.3s ease;
    }
    
    .thumbnail-wrapper:hover .thumbnail-badge {
        background: rgba(102, 126, 234, 0.9);
        transform: scale(1.05);
    }
    
    /* Back Button */
    .back-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #475569;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 2rem;
        text-decoration: none;
    }
    
    .back-btn:hover {
        background: white;
        border-color: #667eea;
        color: #667eea;
        transform: translateX(-8px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Detail Images Grid */
    .detail-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        padding: 1rem 0;
    }
    
    .detail-image-container {
        position: relative;
        overflow: hidden;
        border-radius: 16px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .detail-image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .detail-image {
        width: 100%;
        height: auto;
        display: block;
        transition: all 0.3s ease;
    }
    
    .detail-image-container:hover .detail-image {
        transform: scale(1.05);
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(8px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
        outline: none;
        background: white;
    }
    
    /* Radio Button Styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        backdrop-filter: blur(8px);
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.8);
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        transition: all 0.3s ease;
        backdrop-filter: blur(8px);
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background: white;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
    }
    
    /* Success Message */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 12px;
        border: none;
        color: white;
        font-weight: 500;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .main-content {
            margin: 1rem;
            padding: 1.5rem;
        }
        
        .thumbnail-container {
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
        }
        
        .detail-grid {
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        }
    }
    
    /* Loading Animation */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: .5;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "selected_person" not in st.session_state:
    st.session_state.selected_person = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Load CSS
inject_modern_css()

# Utility functions
def encode_image_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def create_thumbnail_button(person_folder, folder_path, index):
    """Create an enhanced thumbnail button with hover effects"""
    thumbnail_path = os.path.join(folder_path, "thumbnail.jpg")
    if os.path.exists(thumbnail_path):
        image_count = len([f for f in os.listdir(folder_path) if f != "thumbnail.jpg"])
        person_name = f"Person {index + 1}"
        
        # Use Streamlit button with custom styling
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            if st.button(person_name, key=f"btn_{person_folder}", use_container_width=True):
                st.session_state.selected_person = person_folder
                st.session_state.current_page = "person_detail"
                st.rerun()
            
            # Display thumbnail image
            try:
                thumbnail_img = Image.open(thumbnail_path)
                st.image(thumbnail_img, use_container_width=True, caption=f"{image_count} photos")
            except Exception as e:
                st.error(f"Error loading thumbnail: {e}")

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-content">
        <div class="hero-header">
            <h1 class="hero-title">Face Grouping Tool</h1>
            <p class="hero-subtitle">Advanced AI-powered face detection and organization</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if st.session_state.current_page == "home":
        show_home_page()
    elif st.session_state.current_page == "person_detail":
        show_person_detail()
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_home_page():
    """Display the main home page"""
    
    # Source selection
    st.markdown("### Choose Your Image Source")
    
    source_type = st.radio(
        "",
        options=["Google Drive Folder URL", "Local File Upload"],
        horizontal=True,
        disabled=st.session_state.is_processing
    )
    
    if source_type == "Google Drive Folder URL":
        st.markdown("#### Google Drive Integration")
        
        drive_url = st.text_input(
            "Folder URL",
            placeholder="https://drive.google.com/drive/folders/your-folder-id",
            help="Ensure the folder is publicly accessible"
        )
        
        if st.button("Process Images", disabled=st.session_state.is_processing or not drive_url):
            process_google_drive_images(drive_url)
    
    elif source_type == "Local File Upload":
        st.markdown("#### Local File Upload")
        
        uploaded_images = st.file_uploader(
            "Select Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Choose multiple image files (JPG, JPEG, PNG)"
        )
        
        if st.button("Process Images", disabled=st.session_state.is_processing or not uploaded_images):
            process_uploaded_images(uploaded_images)
    
    # Display results
    display_face_groups()

def process_google_drive_images(url):
    """Process images from Google Drive"""
    st.session_state.is_processing = True
    
    try:
        with st.spinner("Downloading images..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_download(fraction):
                progress_bar.progress(fraction)
                status_text.text(f"Downloading: {int(fraction * 100)}%")
            
            download_gdrive_folder(url, DOWNLOAD_DIR, progress_callback=update_download)
        
        with st.spinner("Processing faces..."):
            def update_process(fraction):
                progress_bar.progress(fraction)
                status_text.text(f"Processing: {int(fraction * 100)}%")
            
            run_pipeline(DOWNLOAD_DIR, OUTPUT_DIR, update_progress=update_process)
        
        progress_bar.empty()
        status_text.empty()
        st.success("ðŸŽ‰ Processing complete!")
        
    except Exception as e:
        st.error(f"Error processing images: {str(e)}")
    finally:
        st.session_state.is_processing = False
        st.rerun()

def process_uploaded_images(files):
    """Process uploaded image files"""
    st.session_state.is_processing = True
    
    try:
        # Create directory
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        
        # Save files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(files):
            file_path = os.path.join(DOWNLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            progress = (i + 1) / len(files)
            progress_bar.progress(progress * 0.3)  # 30% for file saving
            status_text.text(f"Saving files: {i + 1}/{len(files)}")
        
        # Process faces
        status_text.text("Detecting and grouping faces...")
        
        def update_process(fraction):
            # Use remaining 70% of progress bar
            progress_bar.progress(0.3 + (fraction * 0.7))
            status_text.text(f"Processing: {int((0.3 + fraction * 0.7) * 100)}%")
        
        run_pipeline(DOWNLOAD_DIR, OUTPUT_DIR, update_progress=update_process)
        
        progress_bar.empty()
        status_text.empty()
        st.success("ðŸŽ‰ Processing complete!")
        
    except Exception as e:
        st.error(f"Error processing images: {str(e)}")
    finally:
        st.session_state.is_processing = False
        st.rerun()

def display_face_groups():
    """Display detected face groups with enhanced thumbnails"""
    if not os.path.exists(OUTPUT_DIR):
        return
    
    # Get person folders
    person_folders = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("person_")]
    
    if not person_folders:
        return
    
    # Sort by number of images (descending)
    person_folders.sort(
        key=lambda x: len([f for f in os.listdir(os.path.join(OUTPUT_DIR, x)) if f != "thumbnail.jpg"]),
        reverse=True
    )
    
    st.markdown(f"""
    <div class="thumbnail-section">
        <h2 class="section-title">Detected People</h2>
        <p class="section-subtitle">Found {len(person_folders)} unique individuals. Click on any person to view all their photos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create thumbnail grid
    cols_per_row = 5
    for i in range(0, len(person_folders), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, person_folder in enumerate(person_folders[i:i + cols_per_row]):
            if j < len(cols):
                with cols[j]:
                    create_thumbnail_button(person_folder, os.path.join(OUTPUT_DIR, person_folder), i + j)

def show_person_detail():
    """Display detailed view for selected person"""
    
    # Back button
    if st.button("â† Back to Overview", key="back_to_home"):
        st.session_state.current_page = "home"
        st.session_state.selected_person = None
        st.rerun()
    
    if not st.session_state.selected_person:
        st.error("No person selected")
        return
    
    person_folder = st.session_state.selected_person
    person_path = os.path.join(OUTPUT_DIR, person_folder)
    
    if not os.path.exists(person_path):
        st.error("Person folder not found")
        return
    
    # Get all images except thumbnail
    images = [f for f in os.listdir(person_path) if f != "thumbnail.jpg"]
    person_name = person_folder.replace("person_", "Person ")
    
    st.markdown(f"### {person_name}")
    st.markdown(f"**{len(images)}** photos found")
    
    # Display images in responsive grid
    if images:
        cols_per_row = 4
        for i in range(0, len(images), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, img_name in enumerate(images[i:i + cols_per_row]):
                if j < len(cols):
                    with cols[j]:
                        img_path = os.path.join(person_path, img_name)
                        try:
                            image = Image.open(img_path)
                            st.image(image, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image {img_name}: {e}")
    else:
        st.warning("No images found for this person")

if __name__ == "__main__":
    main()