import json
import logging
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Clusterer:
    def __init__(self, vector_store, n_clusters=5):
        """
        Initialize the Clusterer.
        
        Args:
            vector_store: The VectorStore instance to get embeddings from
            n_clusters: Number of clusters to create (default: 5)
        """
        self.vector_store = vector_store
        self.n_clusters = n_clusters
        
    def get_clusters(self):
        """
        Retrieve all embeddings from the vector store and cluster them.
        
        Returns:
            dict: A dictionary mapping cluster IDs to lists of filenames
        """
        try:
            # Get all embeddings and metadata from vector store
            logger.info(f"Retrieving all embeddings from vector store")
            data = self.vector_store.get_all_data()
            
            if not data or len(data) < self.n_clusters:
                logger.warning(f"Not enough data for {self.n_clusters} clusters. Only {len(data) if data else 0} embeddings found.")
                return {}
            
            # Extract vectors and file info
            vectors = np.array([item['vector'] for item in data])
            metadatas = [json.loads(item['metadata']) for item in data]
            
            # Perform K-means clustering
            logger.info(f"Performing K-means clustering with {self.n_clusters} clusters")
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(vectors)
            
            # Group files by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                file_path = metadatas[i].get('file', 'unknown')
                clusters[int(label)].append(file_path)
            
            # Remove duplicates from each cluster
            for label in clusters:
                clusters[label] = list(set(clusters[label]))
            
            logger.info(f"Clustering complete. Found {len(clusters)} clusters.")
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {}
            
    def print_clusters(self):
        """
        Print the clusters in a formatted way to the console.
        """
        clusters = self.get_clusters()
        
        if not clusters:
            print("No clusters found. Please ensure your vector database contains indexed files.")
            return
            
        print("\n===== File Clusters (Similar Functionality) =====\n")
        
        for cluster_id, files in clusters.items():
            print(f"Cluster {cluster_id + 1}:")
            for file in files:
                print(f"  - {file}")
            print() 