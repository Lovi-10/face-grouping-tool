import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .config import FACE_SIZE

face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def detect_faces(image):
    return face_app.get(image)

def extract_face_embedding(face):
    return face.normed_embedding


def crop_face(face, image, size=(112, 112)):
    try:
        x1, y1, x2, y2 = map(int, face.bbox)

        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        return cv2.resize(face_crop, size)

    except Exception as e:
        print(f"⚠️ Error cropping face: {e}")
        return None
# def crop_face(face, image):
#     x1, y1, x2, y2 = map(int, face.bbox)
#     return cv2.resize(image[y1:y2, x1:x2], FACE_SIZE)