#!/usr/bin/env python3
import argparse
import os
import sys
import logging
from pathlib import Path
import textwrap

from embedder import SentenceTransformersEmbedder
from vector_store import VectorStore
from searcher import Searcher
from clustering import Clusterer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_embedder(model_name):
    """Initialize the embedder with the specified model."""
    try:
        logger.info(f"Setting up embedder with model: {model_name}")
        return SentenceTransformersEmbedder(model_name)
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        sys.exit(1)

def setup_vector_store(db_path, dimension):
    """Initialize the vector store with the specified database path."""
    try:
        logger.info(f"Setting up vector store at: {db_path}")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return VectorStore(db_path, dimension)
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        sys.exit(1)

def format_result(result, max_chars=5000):
    """Format a search result, truncating text if necessary."""
    text = result.get("text", "")
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    
    return {
        "file": result.get("file", ""),
        "lines": result.get("lines", ""),
        "function": result.get("function", ""),
        "text": text
    }

def interactive_search(searcher, top_k=5):
    """Run an interactive search session."""
    print("\n=== Semantic Code Search ===")
    print("Type 'exit' or 'quit' to end the session")
    print(f"Results limited to top {top_k} matches\n")
    
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query:
                print("Please enter a valid query")
                continue
            
            results = searcher.search(query, top_k=top_k)
            
            if not results:
                print("No results found")
                continue
            
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                formatted = format_result(result)
                print(f"\n--- Result {i} ---")
                print(f"File: {formatted['file']}")
                print(f"Lines: {formatted['lines']}")
                if formatted['function']:
                    print(f"Function: {formatted['function']}")
                print("\nCode:")
                print(textwrap.indent(formatted['text'], '    '))
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error during search: {e}")
            print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Semantic Code Search CLI")
    parser.add_argument(
        "--db-path", 
        type=str, 
        default="./vector_store.db",
        help="Path to the vector database"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qodo/Qodo-Embed-1-1.5B",
        help="Name of the embedding model to use"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5,
        help="Number of results to return for each search"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--clusters",
        action="store_true",
        help="Show file clusters based on semantic similarity"
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=5,
        help="Number of clusters to create when using --clusters (default: 5)"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup components
    embedder = setup_embedder(args.model)
    vector_store = setup_vector_store(args.db_path, embedder.get_dimension())
    
    # If clustering is requested, show clusters and exit
    if args.clusters:
        clusterer = Clusterer(vector_store, n_clusters=args.num_clusters)
        clusterer.print_clusters()
        return
    
    # Otherwise continue with interactive search as before
    searcher = Searcher(embedder, vector_store)
    interactive_search(searcher, args.top_k)

if __name__ == "__main__":
    main() 