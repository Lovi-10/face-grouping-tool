# grouper.py - Enhanced with better clustering parameters and post-processing
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return 1 - cosine(embedding1, embedding2)

def cluster_faces(embeddings, eps=0.6, min_samples=1, merge_threshold=0.7):
    """
    Enhanced face clustering with better parameters and post-processing.
    
    Args:
        embeddings: List of face embeddings
        eps: DBSCAN epsilon parameter (increased from 0.5 to 0.6 for better grouping)
        min_samples: Minimum samples per cluster (kept at 1 for face clustering)
        merge_threshold: Threshold for merging similar clusters (0.7 = 70% similarity)
    
    Returns:
        Array of cluster labels with post-processing applied
    """
    if not embeddings:
        return []
    
    embeddings_array = np.array(embeddings)
    
    # Step 1: Initial DBSCAN clustering with relaxed parameters
    logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)
    initial_labels = clustering.fit_predict(embeddings_array)
    
    logger.info(f"Initial clustering: {len(set(initial_labels))} clusters found")
    
    # Step 2: Post-processing to merge similar clusters
    merged_labels = merge_similar_clusters(embeddings_array, initial_labels, merge_threshold)
    
    final_cluster_count = len(set(merged_labels))
    logger.info(f"After merging: {final_cluster_count} final clusters")
    
    return merged_labels

def merge_similar_clusters(embeddings, labels, threshold=0.7):
    """
    Post-processing step to merge clusters that are very similar.
    This helps fix cases where the same person was split into multiple clusters.
    
    Args:
        embeddings: Array of face embeddings
        labels: Initial cluster labels from DBSCAN
        threshold: Similarity threshold for merging clusters (0.7 = 70% similarity)
    
    Returns:
        Updated cluster labels with similar clusters merged
    """
    unique_labels = list(set(labels))
    if len(unique_labels) <= 1:
        return labels
    
    # Calculate cluster centroids (average embeddings per cluster)
    cluster_centroids = {}
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_embeddings = embeddings[labels == label]
        cluster_centroids[label] = np.mean(cluster_embeddings, axis=0)
    
    # Find pairs of clusters that should be merged
    merge_map = {}  # Maps old label to new label
    processed_labels = set()
    
    for i, label1 in enumerate(cluster_centroids.keys()):
        if label1 in processed_labels:
            continue
            
        merge_group = [label1]
        
        for label2 in list(cluster_centroids.keys())[i+1:]:
            if label2 in processed_labels:
                continue
                
            # Calculate similarity between cluster centroids
            similarity = calculate_cosine_similarity(
                cluster_centroids[label1], 
                cluster_centroids[label2]
            )
            
            if similarity >= threshold:
                merge_group.append(label2)
                logger.info(f"Merging clusters {label1} and {label2} (similarity: {similarity:.3f})")
        
        # Assign all labels in merge group to the first label
        target_label = merge_group[0]
        for label in merge_group:
            merge_map[label] = target_label
            processed_labels.add(label)
    
    # Apply merge mapping to all labels
    merged_labels = []
    for label in labels:
        if label == -1:  # Keep noise points as-is
            merged_labels.append(label)
        else:
            merged_labels.append(merge_map.get(label, label))
    
    return np.array(merged_labels)

def threshold_based_clustering(embeddings, threshold=0.6):
    """
    Alternative clustering approach using threshold-based grouping.
    Often more accurate than DBSCAN for face clustering.
    
    Args:
        embeddings: List of face embeddings
        threshold: Similarity threshold (0.6 = 60% similarity required)
    
    Returns:
        Array of cluster labels
    """
    if not embeddings:
        return []
    
    embeddings_array = np.array(embeddings)
    n_embeddings = len(embeddings_array)
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((n_embeddings, n_embeddings))
    for i in range(n_embeddings):
        for j in range(i+1, n_embeddings):
            sim = calculate_cosine_similarity(embeddings_array[i], embeddings_array[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim
    
    # Group faces based on threshold
    labels = [-1] * n_embeddings  # Start with all as noise
    current_cluster = 0
    
    for i in range(n_embeddings):
        if labels[i] != -1:  # Already assigned
            continue
            
        # Start new cluster
        labels[i] = current_cluster
        
        # Find all similar faces
        for j in range(i+1, n_embeddings):
            if labels[j] == -1 and similarity_matrix[i][j] >= threshold:
                labels[j] = current_cluster
        
        current_cluster += 1
    
    logger.info(f"Threshold clustering: {current_cluster} clusters found")
    return np.array(labels)

def adaptive_clustering(embeddings, initial_eps=0.6, merge_threshold=0.7):
    """
    Adaptive clustering that tries both DBSCAN and threshold-based approaches
    and chooses the better result based on cluster quality metrics.
    
    Args:
        embeddings: List of face embeddings
        initial_eps: Starting epsilon for DBSCAN
        merge_threshold: Threshold for post-processing merge
        
    Returns:
        Best cluster labels found
    """
    if not embeddings:
        return []
    
    # Try DBSCAN with post-processing
    dbscan_labels = cluster_faces(embeddings, eps=initial_eps, merge_threshold=merge_threshold)
    
    # Try threshold-based clustering
    threshold_labels = threshold_based_clustering(embeddings, threshold=0.6)
    
    # Simple quality metric: prefer fewer clusters with similar cluster counts
    dbscan_clusters = len(set(dbscan_labels))
    threshold_clusters = len(set(threshold_labels))
    
    logger.info(f"DBSCAN result: {dbscan_clusters} clusters")
    logger.info(f"Threshold result: {threshold_clusters} clusters")
    
    # Choose the approach that creates fewer clusters (less over-segmentation)
    if threshold_clusters < dbscan_clusters:
        logger.info("Using threshold-based clustering result")
        return threshold_labels
    else:
        logger.info("Using DBSCAN clustering result")
        return dbscan_labels