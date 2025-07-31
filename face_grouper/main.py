import os
import cv2
from .config import IMAGE_EXTENSIONS
from .detector import detect_faces, extract_face_embedding
from .grouper import cluster_faces
from .organizer import organize_photos

def load_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return paths

def process_images(source_folder):
    embeddings, photo_data = [], []
    image_paths = load_images(source_folder)

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        faces = detect_faces(image)
        for face in faces:
            emb = extract_face_embedding(face)
            embeddings.append(emb)
            photo_data.append((path, face))

    return embeddings, photo_data

def run_pipeline(source_folder, output_folder):
    embeddings, photo_data = process_images(source_folder)
    labels = cluster_faces(embeddings)
    return organize_photos(photo_data, labels, output_folder)