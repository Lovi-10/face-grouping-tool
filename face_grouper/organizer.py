import os
import shutil
import cv2
from collections import defaultdict
from .logger import get_logger
from .detector import crop_face

logger = get_logger(__name__)

def crop_face(bbox, image, size=(100, 100)):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    crop = image[y:y+h, x:x+w]
    if crop.size == 0:
        return None

    # Center crop to square
    h_, w_ = crop.shape[:2]
    min_dim = min(h_, w_)
    start_x = (w_ - min_dim) // 2
    start_y = (h_ - min_dim) // 2
    square = crop[start_y:start_y+min_dim, start_x:start_x+min_dim]

    return cv2.resize(square, size)

def handle_no_faces(no_face_paths, output_folder):
    no_face_dir = os.path.join(output_folder, "no_faces_found")
    os.makedirs(no_face_dir, exist_ok=True)

    for i, path in enumerate(no_face_paths):
        filename = os.path.basename(path)
        dst = os.path.join(no_face_dir, f"{i}_{filename}")
        shutil.copy2(path, dst)


def organize_photos(photo_data, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    grouped = defaultdict(list)

    for data, label in zip(photo_data, labels):
        grouped[label].append(data)

    sorted_groups = sorted(grouped.items(), key=lambda x: -len(x[1]))

    for i, (label, items) in enumerate(sorted_groups):
        group_folder = os.path.join(output_dir, f'person_{i+1}')
        thumb_path = os.path.join(group_folder, 'thumbnail.jpg')
        os.makedirs(group_folder, exist_ok=True)

        # Copy all images
        for j, (img_path, face) in enumerate(items):
            filename = os.path.basename(img_path)
            dst = os.path.join(group_folder, f'{j}_{filename}')
            shutil.copy2(img_path, dst)

        best_crop = None
        best_score = -1

        for img_path, face in items:
            image = cv2.imread(img_path)
            if image is None or not hasattr(face, "bbox"):
                continue

            # Proper bbox extraction from insightface
            x1, y1, x2, y2 = face.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            area = w * h

            crop = crop_face((x1, y1, w, h), image)

            # Validate crop
            if crop is not None and crop.size > 0:
                # Optionally: Add more scoring logic here (sharpness, face center)
                score = area  # Simple: use face area as score
                if score > best_score:
                    best_crop = crop
                    best_score = score

        if best_crop is not None:
            cv2.imwrite(thumb_path, best_crop)
        else:
            logger.warning(f"No valid thumbnail found for group {label}")

    return sorted_groups