# organizer.py - Simple crop-and-resize approach (no padding)
import os
import shutil
import cv2
import numpy as np
from collections import defaultdict
from .logger import get_logger
from .detector import crop_face, calculate_face_quality_score

logger = get_logger(__name__)

def handle_no_faces(no_face_paths, output_folder):
    """Handle images where no faces were detected."""
    no_face_dir = os.path.join(output_folder, "no_faces_found")
    os.makedirs(no_face_dir, exist_ok=True)

    for i, path in enumerate(no_face_paths):
        filename = os.path.basename(path)
        dst = os.path.join(no_face_dir, f"{i}_{filename}")
        shutil.copy2(path, dst)

def create_fallback_thumbnail(items, group_folder, thumbnail_size=(150, 150)):
    """
    Create a fallback thumbnail from the first available image if quality-based selection fails.
    
    Args:
        items: List of (img_path, face) tuples
        group_folder: Path to group folder
        thumbnail_size: Size of thumbnail as (width, height) - default (150, 150)
        
    Returns:
        True if thumbnail was created, False otherwise
    """
    thumb_path = os.path.join(group_folder, 'thumbnail.jpg')
    
    for img_path, face in items:
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # Use the simple crop function to ensure exact size
            cropped_face = crop_face(face, image, size=thumbnail_size)
            
            if cropped_face is not None and cropped_face.size > 0:
                cv2.imwrite(thumb_path, cropped_face)
                logger.info(f"Created fallback thumbnail ({thumbnail_size[0]}x{thumbnail_size[1]}) for group from {os.path.basename(img_path)}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to create fallback thumbnail from {img_path}: {e}")
            continue
    
    return False

def select_best_thumbnail(items, group_folder, thumbnail_size=(150, 150)):
    """
    Select the best face for thumbnail based on comprehensive quality scoring.
    All thumbnails will be exactly thumbnail_size[0] x thumbnail_size[1] pixels.
    
    Args:
        items: List of (img_path, face) tuples
        group_folder: Path to group folder
        thumbnail_size: Size of thumbnail as (width, height) - default (150, 150)
        
    Returns:
        True if thumbnail was created successfully
    """
    thumb_path = os.path.join(group_folder, 'thumbnail.jpg')
    
    best_crop = None
    best_score = -1
    best_image_info = None
    
    # Evaluate all faces in the group
    for img_path, face in items:
        try:
            image = cv2.imread(img_path)
            if image is None or not hasattr(face, "bbox"):
                continue
            
            # Calculate comprehensive quality score
            quality_score = calculate_face_quality_score(face, image)
            
            # Get the cropped face with exact dimensions
            cropped_face = crop_face(face, image, size=thumbnail_size)
            
            if cropped_face is not None and cropped_face.size > 0:
                if quality_score > best_score:
                    best_crop = cropped_face.copy()
                    best_score = quality_score
                    best_image_info = img_path
                    
        except Exception as e:
            logger.warning(f"Error processing face from {img_path}: {e}")
            continue
    
    # Save the best thumbnail
    if best_crop is not None:
        try:
            cv2.imwrite(thumb_path, best_crop)
            logger.info(f"Created quality-based thumbnail ({thumbnail_size[0]}x{thumbnail_size[1]}) for group from {os.path.basename(best_image_info)} (score: {best_score:.3f})")
            return True
        except Exception as e:
            logger.error(f"Failed to save thumbnail: {e}")
            return False
    
    return False

def create_placeholder_thumbnail(group_folder, thumbnail_size=(150, 150)):
    """
    Create a placeholder thumbnail as absolute last resort.
    
    Args:
        group_folder: Path to group folder
        thumbnail_size: Size of thumbnail as (width, height) - default (150, 150)
        
    Returns:
        True if placeholder was created successfully
    """
    placeholder_path = os.path.join(group_folder, 'thumbnail.jpg')
    
    try:
        width, height = thumbnail_size
        
        # Create a simple placeholder image (exact dimensions)
        placeholder = np.full((height, width, 3), 128, dtype=np.uint8)  # Gray background
        
        # Add text to indicate no thumbnail
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(width, height) / 300)  # Scale font based on thumbnail size
        thickness = max(1, int(min(width, height) / 100))  # Scale thickness
        
        # Calculate text positions for centering
        text1 = "No"
        text2 = "Thumb"
        
        (text1_w, text1_h), _ = cv2.getTextSize(text1, font, font_scale, thickness)
        (text2_w, text2_h), _ = cv2.getTextSize(text2, font, font_scale, thickness)
        
        # Center the text
        x1 = (width - text1_w) // 2
        y1 = (height - text1_h) // 2 - 5
        x2 = (width - text2_w) // 2  
        y2 = (height + text2_h) // 2 + 5
        
        cv2.putText(placeholder, text1, (x1, y1), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(placeholder, text2, (x2, y2), font, font_scale, (255, 255, 255), thickness)
        
        cv2.imwrite(placeholder_path, placeholder)
        logger.info(f"Created placeholder thumbnail ({width}x{height}) for group")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create placeholder thumbnail: {e}")
        return False

def organize_photos(photo_data, labels, output_dir, thumbnail_size=(150, 150)):
    """
    Organize photos by face clusters with guaranteed consistent thumbnail generation.
    All thumbnails will be exactly thumbnail_size[0] x thumbnail_size[1] pixels.
    
    Args:
        photo_data: List of (img_path, face) tuples
        labels: Cluster labels for each face
        output_dir: Output directory path
        thumbnail_size: Size of thumbnails as (width, height) - default (150, 150)
        
    Returns:
        List of (label, items) tuples sorted by group size
    """
    os.makedirs(output_dir, exist_ok=True)
    grouped = defaultdict(list)

    # Group faces by cluster labels
    for data, label in zip(photo_data, labels):
        grouped[label].append(data)

    # Sort groups by size (largest first)
    sorted_groups = sorted(grouped.items(), key=lambda x: -len(x[1]))

    for i, (label, items) in enumerate(sorted_groups):
        group_folder = os.path.join(output_dir, f'person_{i+1}')
        os.makedirs(group_folder, exist_ok=True)

        # Copy all images to the group folder
        for j, (img_path, face) in enumerate(items):
            try:
                filename = os.path.basename(img_path)
                dst = os.path.join(group_folder, f'{j}_{filename}')
                shutil.copy2(img_path, dst)
            except Exception as e:
                logger.warning(f"Failed to copy image {img_path}: {e}")

        # GUARANTEED THUMBNAIL CREATION (3-tier system)
        thumbnail_created = False
        width, height = thumbnail_size
        
        # Tier 1: Quality-based thumbnail selection
        logger.info(f"Creating thumbnail for person_{i+1} using quality-based selection...")
        thumbnail_created = select_best_thumbnail(items, group_folder, thumbnail_size)
        
        # Tier 2: Simple fallback if quality selection fails
        if not thumbnail_created:
            logger.warning(f"Quality-based thumbnail selection failed for group {label}, trying fallback...")
            thumbnail_created = create_fallback_thumbnail(items, group_folder, thumbnail_size)
        
        # Tier 3: Placeholder as absolute last resort
        if not thumbnail_created:
            logger.warning(f"All thumbnail creation methods failed for group {label}, creating placeholder...")
            thumbnail_created = create_placeholder_thumbnail(group_folder, thumbnail_size)
        
        # Final verification
        if thumbnail_created:
            thumb_path = os.path.join(group_folder, 'thumbnail.jpg')
            if os.path.exists(thumb_path):
                # Verify thumbnail has correct dimensions
                thumb_img = cv2.imread(thumb_path)
                if thumb_img is not None:
                    actual_h, actual_w = thumb_img.shape[:2]
                    if actual_h == height and actual_w == width:
                        logger.info(f"âœ… Verified thumbnail for person_{i+1}: {width}x{height}")
                    else:
                        logger.warning(f"âš ï¸  Thumbnail size mismatch for person_{i+1}: {actual_w}x{actual_h} (expected {width}x{height})")
                else:
                    logger.error(f"âŒ Thumbnail file corrupted for person_{i+1}")
            else:
                logger.error(f"âŒ Thumbnail file missing for person_{i+1}")
        else:
            logger.error(f"âŒ CRITICAL: Failed to create any thumbnail for group {label}")

    return sorted_groups