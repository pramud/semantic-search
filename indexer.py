import os
import logging

logging.basicConfig(level=logging.INFO)

class Indexer:
    def __init__(self, embedder, vector_store, chunker_registry):
        logging.info("Initializing Indexer")
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker_registry = chunker_registry
        
        # Directories to skip when indexing
        self.skip_dirs = {
            '.git',           # Git version control
            '.github',        # GitHub Actions and configuration
            '.hg',            # Mercurial version control
            '.svn',           # Subversion version control
            'node_modules',   # npm dependencies
            '__pycache__',    # Python bytecode
            'venv',           # Python virtual environments
            '.venv',
            'env',
            'build',          # Build artifacts
            'dist',           # Distribution packages
            'out',            # Output folders
            'target',         # Maven/Rust build output
            'bin',            # Binary outputs
            'obj',            # Object files
            '.idea',          # JetBrains IDE files
            '.vscode'         # VS Code settings
        }
        
        # File extensions to skip
        self.skip_extensions = {
            '.log',           # Log files
            '.pyc',           # Python bytecode
            '.class',         # Java bytecode
            '.o',             # Object files
            '.so',            # Shared libraries
            '.dll',           # Windows libraries
            '.exe',           # Executables
            '.bin',           # Binary files
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',  # Images
            '.pdf', '.doc', '.docx', '.ppt', '.pptx',         # Documents
            '.zip', '.tar', '.gz', '.7z', '.rar',             # Archives
        }

    def index_directory(self, dir_path):
        logging.info(f"Starting to index directory: {dir_path}")
        file_count = 0
        skipped_count = 0
        
        for root, dirs, files in os.walk(dir_path):
            # Modify dirs in-place to skip certain directories
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            # Skip hidden directories (starting with .)
            dirs[:] = [d for d in dirs if not d.startswith('.') or d in ('.github',)]
            
            rel_path = os.path.relpath(root, dir_path)
            if rel_path != '.' and any(part in self.skip_dirs for part in rel_path.split(os.sep)):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip files with specific extensions
                _, ext = os.path.splitext(file_path)
                if ext.lower() in self.skip_extensions:
                    logging.debug(f"Skipping file with excluded extension: {file_path}")
                    skipped_count += 1
                    continue
                    
                # Skip hidden files (starting with .)
                if file.startswith('.'):
                    logging.debug(f"Skipping hidden file: {file_path}")
                    skipped_count += 1
                    continue
                
                logging.info(f"Processing file: {file_path}")
                try:
                    file_type = self.chunker_registry.determine_file_type(file_path)
                    logging.debug(f"File type determined: {file_type}")
                    chunker = self.chunker_registry.get_chunker(file_type)
                    if not chunker:
                        logging.warning(f"No chunker available for {file_path}")
                        # Try using generic chunker as fallback
                        chunker = self.chunker_registry.get_chunker("generic")
                        if not chunker:
                            logging.warning(f"No generic chunker available, skipping {file_path}")
                            skipped_count += 1
                            continue
                    
                    chunks = chunker.chunk_file(file_path)
                    logging.info(f"Extracted {len(chunks)} chunks from {file_path}")
                    for chunk in chunks:
                        # Parse the metadata from the chunk header
                        lines = chunk.split('\n', 3)
                        if len(lines) < 4:
                            logging.warning(f"Invalid chunk format in {file_path}, skipping")
                            continue
                            
                        metadata = {
                            "file": lines[0].replace("File: ", ""),
                            "function": lines[1].replace("Function: ", ""),
                            "lines": lines[2].replace("Lines: ", ""),
                            "text": lines[3]
                        }
                        logging.debug(f"Embedding chunk from {file_path}: {metadata['text'][:50]}...")
                        embedding = self.embedder.embed(metadata["text"])
                        self.vector_store.insert(embedding, metadata)
                        logging.debug(f"Inserted chunk into vector store")
                    
                    file_count += 1
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {e}")
                    skipped_count += 1
        
        logging.info(f"Finished indexing directory: {dir_path}")
        logging.info(f"Processed {file_count} files, skipped {skipped_count} files")

    def determine_file_type(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return {"cpp": "cpp", "hpp": "cpp", "py": "py"}.get(ext[1:], None)