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
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic code search tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Search for 'file handling' in the default vector database
              semantic-search "file handling"
              
              # Cluster files with similar functionality
              semantic-search --cluster
        """)
    )
    
    parser.add_argument("query", nargs="?", default="", help="Search query (optional for clustering)")
    parser.add_argument("--db", default="./codebase_vectors", help="Vector database path")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--cluster", action="store_true", help="Display clusters of similar files")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters to create")
    
    args = parser.parse_args()
    
    # Set up components
    embedder = setup_embedder(args.model)
    dimension = embedder.get_dimension()
    vector_store = setup_vector_store(args.db, dimension)
    
    if args.cluster:
        # Run clustering functionality
        logger.info(f"Running clustering with {args.n_clusters} clusters")
        clusterer = Clusterer(vector_store, args.n_clusters)
        clusterer.print_clusters()
    elif args.query:
        # Run search functionality
        searcher = Searcher(embedder, vector_store)
        results = searcher.search(args.query, args.top_k)
        
        if not results:
            print("No results found.")
            return
            
        print(f"\nSearch results for: '{args.query}'\n")
        for i, result in enumerate(results):
            formatted = format_result(result)
            print(f"Result {i+1}:")
            print(f"  File: {formatted['file']}")
            print(f"  Lines: {formatted['lines']}")
            print(f"  Function: {formatted['function']}")
            print("  Snippet:")
            for line in formatted['text'].split('\n'):
                print(f"    {line}")
            print()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
