from sklearn.cluster import DBSCAN
import numpy as np

def cluster_faces(embeddings):
    if not embeddings:
        return []
    clustering = DBSCAN(metric='cosine', eps=0.5, min_samples=1)
    return clustering.fit_predict(np.array(embeddings))