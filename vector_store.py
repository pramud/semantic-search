import lancedb
import pyarrow as pa
import json
import logging

logging.basicConfig(level=logging.DEBUG)

class VectorStore:
    def __init__(self, db_path, dimension):
        logging.info(f"Initializing VectorStore with db_path: {db_path}, dimension: {dimension}")
        self.db = lancedb.connect(db_path)
        schema = pa.schema([
            pa.field("vector", lancedb.vector(dimension)),
            pa.field("metadata", pa.string())
        ])
        self.table = self.db.create_table("code_chunks", schema=schema, exist_ok=True)

    def insert(self, vector, metadata):
        try:
            logging.debug(f"Inserting vector with metadata: {metadata}")
            self.table.add([{"vector": vector, "metadata": json.dumps(metadata)}])
            logging.debug("Vector inserted successfully")
        except Exception as e:
            logging.error(f"Failed to insert vector: {e}")

    def search(self, query_vector, top_k=5):
        try:
            logging.debug(f"Searching with query vector of length: {len(query_vector)}")
            results = self.table.search(query_vector).limit(top_k).to_pandas()
            logging.info(f"Search returned {len(results)} results")
            return [json.loads(md) for md in results["metadata"].tolist()]
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []

    def get_all_data(self):
        """Retrieve all vector embeddings and metadata from the vector store.
        
        Returns:
            list: A list of dictionaries containing 'vector' and 'metadata' fields
        """
        try:
            logging.info("Retrieving all data from vector store")
            df = self.table.to_pandas()
            logging.info(f"Retrieved {len(df)} records from vector store")
            
            result = []
            for _, row in df.iterrows():
                result.append({
                    'vector': row['vector'],
                    'metadata': row['metadata']
                })
            return result
        except Exception as e:
            logging.error(f"Failed to retrieve all data: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize VectorStore with a sample dimension
    dimension = 384  # Matches all-MiniLM-L6-v2 from SentenceTransformersEmbedder
    vector_store = VectorStore("./codebase_vectors", dimension)

    # Insert a sample embedding
    sample_vector = [0.1] * dimension  # Dummy vector for testing
    sample_metadata = {
        "file": "example.cpp",
        "function": "main",
        "lines": "10-20",
        "text": "int main() {\n    return 0;\n}"
    }
    vector_store.insert(sample_vector, sample_metadata)

    # Search with a query vector
    query_vector = [0.1] * dimension  # Dummy query vector
    results = vector_store.search(query_vector, top_k=1)
    print("Search results:", results)