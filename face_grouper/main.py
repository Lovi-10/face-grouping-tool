import os
import cv2
from .config import IMAGE_EXTENSIONS
from .detector import detect_faces, extract_face_embedding
from .grouper import cluster_faces
<<<<<<< HEAD
from .organizer import organize_photos
=======
from .organizer import organize_photos, handle_no_faces

>>>>>>> origin/staging

def load_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return paths

<<<<<<< HEAD
def process_images(source_folder):
    embeddings, photo_data = [], []
    image_paths = load_images(source_folder)

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        faces = detect_faces(image)
=======
def process_images(source_folder, update_progress=None):
    embeddings, photo_data = [], []
    no_faces = []  # 🆕 List to track images with no faces

    image_paths = load_images(source_folder)
    total = len(image_paths)

    for idx, path in enumerate(image_paths):
        image = cv2.imread(path)
        if image is None:
            continue

        faces = detect_faces(image)
        if not faces:  # 🆕 No faces detected
            no_faces.append(path)
>>>>>>> origin/staging
        for face in faces:
            emb = extract_face_embedding(face)
            embeddings.append(emb)
            photo_data.append((path, face))

<<<<<<< HEAD
    return embeddings, photo_data

def run_pipeline(source_folder, output_folder):
    embeddings, photo_data = process_images(source_folder)
    labels = cluster_faces(embeddings)
    return organize_photos(photo_data, labels, output_folder)
=======
        if update_progress:
            update_progress((idx + 1) / total)

    return embeddings, photo_data, no_faces  # 🆕 return extra


def run_pipeline(source_folder, output_folder, update_progress=None):
    embeddings, photo_data, no_faces = process_images(source_folder, update_progress)
    labels = cluster_faces(embeddings)
    clusters = organize_photos(photo_data, labels, output_folder)
    handle_no_faces(no_faces, output_folder)  # 🆕 Add this line
    return clusters

>>>>>>> origin/staging
