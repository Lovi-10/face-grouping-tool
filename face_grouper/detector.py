# detector.py - Enhanced with face alignment and better embedding extraction  
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .config import FACE_SIZE
import logging

logger = logging.getLogger(__name__)

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

def align_face(image, landmarks, target_size=(112, 112)):
    """
    Align face using landmarks to normalize pose and rotation.
    This significantly improves embedding quality and clustering accuracy.
    
    Args:
        image: Input image containing the face
        landmarks: Facial landmarks from InsightFace detection
        target_size: Target size for aligned face
        
    Returns:
        Aligned face image or None if alignment fails
    """
    try:
        if landmarks is None or len(landmarks) < 5:
            return None
            
        # Use the first 5 landmarks (eyes, nose, mouth corners)
        src_pts = landmarks[:5].astype(np.float32)
        
        # Standard facial landmark positions for alignment (normalized coordinates)
        dst_pts = np.array([
            [30.2946, 51.6963],  # Left eye
            [65.5318, 51.5014],  # Right eye  
            [48.0252, 71.7366],  # Nose tip
            [33.5493, 92.3655],  # Left mouth corner
            [62.7299, 92.2041]   # Right mouth corner
        ], dtype=np.float32)
        
        # Scale destination points to target size
        scale_x = target_size[0] / 96.0
        scale_y = target_size[1] / 112.0
        dst_pts[:, 0] *= scale_x
        dst_pts[:, 1] *= scale_y
        
        # Calculate transformation matrix
        transformation_matrix = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        
        if transformation_matrix is None:
            return None
            
        # Apply alignment transformation
        aligned_face = cv2.warpAffine(image, transformation_matrix, target_size)
        
        return aligned_face
        
    except Exception as e:
        logger.warning(f"Face alignment failed: {e}")
        return None

def extract_enhanced_embedding(face, image):
    """
    Extract high-quality face embedding with alignment preprocessing.
    This improves clustering accuracy by normalizing face pose and lighting.
    
    Args:
        face: InsightFace detection object
        image: Original image
        
    Returns:
        Enhanced face embedding or None if extraction fails
    """
    try:
        # Try face alignment if landmarks are available
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            aligned_face = align_face(image, face.landmark_2d_106)
            if aligned_face is not None:
                # Re-detect face in aligned image for better embedding
                aligned_faces = face_app.get(aligned_face)
                if aligned_faces:
                    aligned_embedding = aligned_faces[0].normed_embedding
                    logger.debug("Using aligned face embedding")
                    return aligned_embedding
        
        # Fallback to regular embedding if alignment fails
        return face.normed_embedding
        
    except Exception as e:
        logger.warning(f"Enhanced embedding extraction failed: {e}")
        return face.normed_embedding

def calculate_face_centredness_score(face_bbox, image_shape):
    """Calculate how centered a face is within the image."""
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
        
        return max(0.0, min(1.0, centredness_score))
        
    except Exception as e:
        logger.warning(f"Error calculating face centredness: {e}")
        return 0.5

def calculate_embedding_quality_score(face, image):
    """
    Calculate comprehensive quality score for face embedding extraction.
    Higher scores indicate better quality for clustering.
    
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
        centredness = calculate_face_centredness_score((x1, y1, x2, y2), image.shape)
        
        # Detection confidence (from InsightFace)
        detection_confidence = getattr(face, 'det_score', 0.5)
        
        # Face pose quality (if landmarks available)
        pose_quality = 1.0
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            # Calculate pose deviation (simplified)
            landmarks = face.landmark_2d_106
            if len(landmarks) >= 5:
                left_eye = landmarks[0]
                right_eye = landmarks[1]
                
                # Calculate eye angle (should be close to horizontal for good pose)
                eye_angle = np.abs(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
                pose_quality = max(0.1, 1.0 - eye_angle / (np.pi / 6))  # Penalize angles > 30 degrees
        
        # Normalize metrics to 0-1 range
        area_score = min(face_area / 10000, 1.0)
        sharpness_score = min(sharpness / 100, 1.0) 
        brightness_score = 1.0 - abs(brightness - 128) / 128
        contrast_score = min(contrast / 64, 1.0)
        
        # ENHANCED WEIGHTED COMBINATION for clustering accuracy
        quality_score = (
            centredness * 0.35 +              # 35% - Face centredness (high priority)
            sharpness_score * 0.25 +          # 25% - Image sharpness
            pose_quality * 0.15 +             # 15% - Face pose quality (NEW)
            detection_confidence * 0.10 +     # 10% - Detection confidence
            area_score * 0.10 +               # 10% - Face size
            contrast_score * 0.025 +          # 2.5% - Contrast
            brightness_score * 0.025          # 2.5% - Brightness
        )
        
        return quality_score
        
    except Exception as e:
        logger.warning(f"Error calculating embedding quality: {e}")
        return 0

def crop_face(face, image, size=(150, 150)):
    """
    Crop face from image and resize to exact dimensions.
    Enhanced version with better error handling.
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
            logger.warning(f"Invalid bounding box: ({x1},{y1},{x2},{y2})")
            return None
        
        # Crop the face region
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            logger.warning("Empty face crop")
            return None
        
        # Resize directly to target dimensions
        target_width, target_height = size
        resized_face = cv2.resize(face_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return resized_face
        
    except Exception as e:
        logger.error(f"Error cropping face: {e}")
        return None

# Alias for backward compatibility
calculate_face_quality_score = calculate_embedding_quality_score