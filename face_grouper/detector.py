# detector.py - Enhanced with face centredness scoring
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

def calculate_image_sharpness(image):
    """Calculate image sharpness using Laplacian variance."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_image_brightness(image):
    """Calculate average brightness of the image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return np.mean(gray)

def calculate_image_contrast(image):
    """Calculate image contrast using standard deviation."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return np.std(gray)

def calculate_face_centredness_score(face_bbox, image_shape):
    """
    Calculate how centered a face is within the image.
    
    Args:
        face_bbox: Face bounding box coordinates (x1, y1, x2, y2)
        image_shape: Image dimensions (height, width, channels)
        
    Returns:
        Centredness score between 0.0 (edge) and 1.0 (perfect center)
    """
    try:
        x1, y1, x2, y2 = face_bbox
        img_height, img_width = image_shape[:2]
        
        # Calculate face center
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        
        # Calculate image center
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        # Calculate distance from face center to image center
        distance = ((face_center_x - img_center_x)**2 + (face_center_y - img_center_y)**2)**0.5
        
        # Calculate maximum possible distance (corner to center)
        max_distance = ((img_width/2)**2 + (img_height/2)**2)**0.5
        
        # Calculate centredness score (1.0 = perfect center, 0.0 = corner)
        centredness_score = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, centredness_score))  # Ensure score is between 0 and 1
        
    except Exception as e:
        print(f"âš ï¸ Error calculating face centredness: {e}")
        return 0.5  # Return neutral score on error

def crop_face(face, image, size=(150, 150)):
    """
    Crop face from image and resize to exact dimensions (may cause distortion).
    This ensures all thumbnails are exactly the same size.
    
    Args:
        face: InsightFace detection object with bbox attribute
        image: Original image (numpy array)
        size: Target dimensions as (width, height) - default (150, 150)
        
    Returns:
        Cropped face image resized to exact dimensions or None if crop fails
    """
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Check if bounding box is valid
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop the face region
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        # Resize directly to target dimensions (ignoring aspect ratio)
        target_width, target_height = size
        resized_face = cv2.resize(face_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return resized_face
        
    except Exception as e:
        print(f"âš ï¸ Error cropping face: {e}")
        return None

def calculate_face_quality_score(face, image):
    """
    Calculate comprehensive quality score for a face with emphasis on centredness.
    
    Args:
        face: InsightFace detection object
        image: Original image
        
    Returns:
        Quality score (higher is better)
    """
    try:
        # Get face bounding box
        x1, y1, x2, y2 = map(int, face.bbox)
        
        # Face area (larger faces generally better)
        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height
        
        # Crop face for quality assessment
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return 0
        
        # Calculate quality metrics
        sharpness = calculate_image_sharpness(face_crop)
        brightness = calculate_image_brightness(face_crop)
        contrast = calculate_image_contrast(face_crop)
        
        # Calculate face centredness score (NEW - MAXIMUM WEIGHTAGE)
        centredness = calculate_face_centredness_score((x1, y1, x2, y2), image.shape)
        
        # Detection confidence (from InsightFace)
        detection_confidence = getattr(face, 'det_score', 0.5)  # Default 0.5 if not available
        
        # Normalize metrics to 0-1 range (approximate)
        area_score = min(face_area / 10000, 1.0)  # Normalize assuming 100x100 as good size
        sharpness_score = min(sharpness / 100, 1.0)  # Normalize sharpness
        brightness_score = 1.0 - abs(brightness - 128) / 128  # Prefer brightness around 128
        contrast_score = min(contrast / 64, 1.0)  # Normalize contrast
        
        # UPDATED WEIGHTED COMBINATION - CENTREDNESS GETS MAXIMUM WEIGHTAGE (50%)
        quality_score = (
            centredness * 0.50 +              # 50% weight for face centredness (MAXIMUM)
            sharpness_score * 0.20 +          # 20% weight for sharpness
            area_score * 0.15 +               # 15% weight for face size
            detection_confidence * 0.10 +     # 10% weight for detection confidence
            brightness_score * 0.025 +        # 2.5% weight for brightness
            contrast_score * 0.025            # 2.5% weight for contrast
        )
        
        return quality_score
        
    except Exception as e:
        print(f"âš ï¸ Error calculating face quality: {e}")
        return 0