#!/usr/bin/env python3
"""
Index pipeline for code semantic search.
This script sets up and runs the indexing pipeline for code files.
"""

import os
import logging
import argparse
import json
from chunker import CppChunker, PythonChunker, TokenBasedChunker
from chunker_registry import ChunkerRegistry
from embedder import SentenceTransformersEmbedder
from vector_store import VectorStore
from indexer import Indexer

max_tokens = 32000
overlap_tokens = 100

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_chunker_registry(languages_lib_path, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
    """Set up the chunker registry with all available chunkers.
    
    Args:
        languages_lib_path (str): Path to the tree-sitter languages library.
        max_tokens (int): Maximum tokens per chunk.
        overlap_tokens (int): Tokens to overlap for context.
        
    Returns:
        ChunkerRegistry: The initialized chunker registry.
    """
    logging.info("Setting up chunker registry")
    chunker_registry = ChunkerRegistry()
    
    # Register C++ chunker
    logging.info(f"Registering C++ chunker with max_tokens={max_tokens}")
    chunker_registry.register_chunker("cpp", CppChunker(languages_lib_path, max_tokens, overlap_tokens))
    
    # Register Python chunker
    logging.info(f"Registering Python chunker with max_tokens={max_tokens}")
    chunker_registry.register_chunker("python", PythonChunker(max_tokens, overlap_tokens))
    
    # Register generic token-based chunker for all other file types
    logging.info(f"Registering generic token-based chunker with max_tokens={max_tokens}")
    chunker_registry.register_chunker("generic", TokenBasedChunker(max_tokens, overlap_tokens))
    
    return chunker_registry

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Index code files for semantic search")
    parser.add_argument("--dir", required=True, help="Directory to index")
    parser.add_argument("--vector-store", default="vector_store.db", help="Path to vector store")
    parser.add_argument("--languages-lib", default="build/languages.dll", help="Path to tree-sitter languages library")
    parser.add_argument("--max-tokens", type=int, default=max_tokens, help="Maximum tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=overlap_tokens, help="Tokens to overlap for context")
    parser.add_argument("--model", default="Qodo/Qodo-Embed-1-1.5B", help="Sentence transformer model to use")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Set logging level")
    parser.add_argument("--skip-dirs", type=str, default=None, 
                        help="Additional directories to skip, comma-separated (e.g., 'tests,docs,examples')")
    parser.add_argument("--include-hidden", action="store_true", 
                        help="Include hidden files and directories (starting with .)")
    parser.add_argument("--skip-extensions", type=str, default=None,
                        help="Additional file extensions to skip, comma-separated (e.g., 'md,txt,json')")
    parser.add_argument("--include-only", type=str, default=None,
                        help="Only include files with these extensions, comma-separated (e.g., 'py,cpp,h')")
    parser.add_argument("--config-file", type=str, default=None,
                        help="JSON configuration file with indexing options")
    args = parser.parse_args()
    
    # Set logging level
    logging_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(logging_level)
    
    # Load config file if provided
    if args.config_file and os.path.exists(args.config_file):
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
                # Update args with config file values
                for key, value in config.items():
                    if not getattr(args, key, None):
                        setattr(args, key, value)
            logging.info(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            logging.error(f"Failed to load config file {args.config_file}: {e}")
    
    # Check if directory exists
    if not os.path.isdir(args.dir):
        logging.error(f"Directory {args.dir} does not exist")
        return
    
    try:
        # Set up components
        logging.info(f"Using model: {args.model}")
        embedder = SentenceTransformersEmbedder(args.model)
        
        logging.info(f"Using vector store: {args.vector_store}")
        vector_store = VectorStore(args.vector_store, embedder.get_dimension())
        
        chunker_registry = setup_chunker_registry(
            args.languages_lib, 
            args.max_tokens, 
            args.overlap_tokens
        )
        
        # Create indexer
        indexer = Indexer(embedder, vector_store, chunker_registry)
        
        # Add additional directories to skip
        if args.skip_dirs:
            skip_dirs = set(args.skip_dirs.split(','))
            logging.info(f"Adding additional directories to skip: {skip_dirs}")
            indexer.skip_dirs.update(skip_dirs)
            
        # Add additional extensions to skip
        if args.skip_extensions:
            skip_exts = set(f".{ext}" for ext in args.skip_extensions.split(','))
            logging.info(f"Adding additional file extensions to skip: {skip_exts}")
            indexer.skip_extensions.update(skip_exts)
            
        # If include-only is specified, clear skip_extensions and only consider the included ones
        if args.include_only:
            include_exts = set(f".{ext}" for ext in args.include_only.split(','))
            logging.info(f"Only including files with extensions: {include_exts}")
            
            # Create a function to check file extensions
            original_index_directory = indexer.index_directory
            
            def filtered_index_directory(dir_path):
                # Save original skip extensions
                original_skip_extensions = indexer.skip_extensions
                
                # Set up a function to check if a file should be included
                def should_include_file(file_path):
                    _, ext = os.path.splitext(file_path)
                    return ext.lower() in include_exts
                
                # Monkey patch the walk function to filter files
                original_walk = os.walk
                def filtered_walk(dir_path):
                    for root, dirs, files in original_walk(dir_path):
                        filtered_files = [f for f in files if should_include_file(os.path.join(root, f))]
                        yield root, dirs, filtered_files
                
                # Temporarily replace os.walk with our filtered version
                os.walk = filtered_walk
                
                # Clear skip extensions
                indexer.skip_extensions = set()
                
                try:
                    # Call the original function
                    original_index_directory(dir_path)
                finally:
                    # Restore original functions and settings
                    os.walk = original_walk
                    indexer.skip_extensions = original_skip_extensions
            
            # Replace the index_directory method with our wrapped version
            indexer.index_directory = filtered_index_directory
            
        # If include-hidden is specified, modify the indexer to include hidden files and directories
        if args.include_hidden:
            logging.info("Including hidden files and directories")
            original_index_directory = indexer.index_directory
            
            def index_with_hidden(dir_path):
                # Save original index_directory method
                original_method = indexer.index_directory
                
                # Monkey patch the walk function to include hidden files/dirs
                original_walk = os.walk
                def walk_with_hidden(dir_path):
                    for root, dirs, files in original_walk(dir_path):
                        # Don't modify dirs, include all directories
                        yield root, dirs, files
                
                # Replace os.walk temporarily
                os.walk = walk_with_hidden
                
                try:
                    # Call original method with modified behavior for hidden files
                    if callable(original_method):
                        if original_method == filtered_index_directory:
                            filtered_index_directory(dir_path)
                        else:
                            original_method(dir_path)
                    else:
                        original_index_directory(dir_path)
                finally:
                    # Restore original function
                    os.walk = original_walk
                    
            # Replace the index_directory method with our wrapped version
            if args.include_only:
                # If include_only is already set, need special handling
                original_filtered_index_directory = filtered_index_directory
                def combined_filtering(dir_path):
                    # Save original os.walk
                    original_walk = os.walk
                    
                    # Create a combined filter
                    def combined_walk(dir_path):
                        for root, dirs, files in original_walk(dir_path):
                            # Don't filter directories for hidden
                            filtered_files = [f for f in files if should_include_file(os.path.join(root, f))]
                            yield root, dirs, filtered_files
                    
                    # Replace os.walk
                    os.walk = combined_walk
                    
                    try:
                        # Call original method
                        original_index_directory(dir_path)
                    finally:
                        # Restore
                        os.walk = original_walk
                
                indexer.index_directory = combined_filtering
            else:
                indexer.index_directory = index_with_hidden
        
        # Run indexing
        logging.info(f"Starting indexing of {args.dir}")
        indexer.index_directory(args.dir)
        logging.info("Indexing complete")
        
    except Exception as e:
        logging.error(f"Error during indexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 