import os
import cv2
import shutil
import numpy as np
from collections import defaultdict
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

# --- Settings ---
image_dir = "input_photos"              # folder containing input images
output_dir = "grouped_faces"      # output directory for sorted folders
others_dir = os.path.join(output_dir, "others")  # for no face detected
os.makedirs(output_dir, exist_ok=True)
os.makedirs(others_dir, exist_ok=True)

# --- Initialize face detector + embedding extractor ---
app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)

embeddings = []
img_paths = []
face_map = []  # face-to-image mapping
no_face_images = []

# --- Step 1: Scan all images ---
for file in os.listdir(image_dir):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(image_dir, file)
    img = cv2.imread(path)
    if img is None:
        continue

    faces = app.get(img)
    if not faces:
        no_face_images.append(path)
        print(f"⚠️ No face in: {file}")
        continue

    for face in faces:
        embeddings.append(face.embedding)
        img_paths.append(path)
        face_map.append(file)

print(f"\n✅ Total faces detected: {len(embeddings)}")

# --- Step 2: Cluster embeddings ---
embeddings_np = np.array(embeddings)
clustering = DBSCAN(eps=0.5, min_samples=1, metric='cosine').fit(embeddings_np)
labels = clustering.labels_

# --- Step 3: Map clusters to unique image paths ---
cluster_to_images = defaultdict(set)
for label, path in zip(labels, img_paths):
    cluster_to_images[label].add(path)

# --- Step 4: Sort clusters by number of photos ---
sorted_clusters = sorted(cluster_to_images.items(), key=lambda item: len(item[1]), reverse=True)

# --- Step 5: Save images in folders sorted by face frequency ---
for i, (cluster_id, paths) in enumerate(sorted_clusters, start=1):
    person_dir = os.path.join(output_dir, f"person_{i:02d}")
    os.makedirs(person_dir, exist_ok=True)

    for p in paths:
        dest = os.path.join(person_dir, os.path.basename(p))
        if not os.path.exists(dest):
            shutil.copy(p, dest)

# --- Step 6: Handle no-face images ---
for path in no_face_images:
    dest = os.path.join(others_dir, os.path.basename(path))
    if not os.path.exists(dest):
        shutil.copy(path, dest)

# --- Summary ---
print(f"\n📁 Grouped images saved under: {output_dir}")
print(f"👥 People grouped: {len(sorted_clusters)}")
print(f"🚫 No-face images stored in: {others_dir} ({len(no_face_images)} files)")