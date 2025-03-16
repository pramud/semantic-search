import logging

logging.basicConfig(level=logging.INFO)

class Searcher:
    def __init__(self, embedder, vector_store):
        logging.info("Initializing Searcher")
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, query, top_k=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        try:
            logging.info(f"Starting search with query: {query}")
            query_embedding = self.embedder.embed(query)
            logging.debug(f"Query embedded, vector length: {len(query_embedding)}")
            results = self.vector_store.search(query_embedding, top_k)
            logging.info(f"Search completed, found {len(results)} results")
            return results
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []