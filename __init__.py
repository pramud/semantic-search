from .embedder import SentenceTransformersEmbedder
from .vector_store import VectorStore
from .chunker import BaseChunker, CppChunker
from .indexer import Indexer
from .searcher import Searcher
from .clustering import Clusterer

class ChunkerRegistry:
    def __init__(self):
        self.chunkers = {}
    
    def register_chunker(self, file_type, chunker):
        self.chunkers[file_type] = chunker
    
    def get_chunker(self, file_type):
        return self.chunkers.get(file_type) 