import os
import shutil
import cv2
from collections import defaultdict
from .logger import get_logger
from .detector import crop_face

logger = get_logger(__name__)

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

        best_crop = None
        max_area = 0

        for j, (img_path, face) in enumerate(items):
            filename = os.path.basename(img_path)
            dst = os.path.join(group_folder, f'{j}_{filename}')
            shutil.copy2(img_path, dst)

            # Safely unpack face coordinates
            if isinstance(face, (list, tuple)) and len(face) >= 4:
                x, y, w, h = face[:4]
            else:
                logger.warning(f"Invalid face format: {face}")
                continue

            area = w * h
            if area > max_area:
                image = cv2.imread(img_path)
                if image is not None:
                    crop = crop_face((x, y, w, h), image)
                    if crop is not None and crop.size > 0:
                        best_crop = crop
                        max_area = area

        if best_crop is not None:
            cv2.imwrite(thumb_path, best_crop)
        else:
            logger.warning(f"No valid thumbnail found for group {label}")

    return sorted_groups