from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

class SentenceTransformersEmbedder(BaseEmbedder):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logging.info(f"Initializing SentenceTransformersEmbedder with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def embed(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")
        try:
            logging.debug(f"Embedding text: {text[:50]}...")
            embedding = self.model.encode(text).tolist()
            logging.debug(f"Generated embedding of length: {len(embedding)}")
            return embedding
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            raise RuntimeError(f"Embedding failed: {e}")

    def get_dimension(self):
        return self.model.get_sentence_embedding_dimension()
    
    
# Example usage
if __name__ == "__main__":
    embedder = SentenceTransformersEmbedder()
    print(f"Dimension: {embedder.get_dimension()}")
    embedding = embedder.embed("Hello, world!")
    print(f"Embedding length: {len(embedding)}")