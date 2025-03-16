import logging
from chunker import TokenBasedChunker, CppChunker, PythonChunker

class ChunkerRegistry:
    """Registry for chunker instances.
    
    This class maintains a registry of chunkers for different file types
    and provides methods to register and retrieve them.
    """
    
    def __init__(self):
        """Initialize an empty chunker registry."""
        self.chunkers = {}
        logging.info("Initialized ChunkerRegistry")
    
    def register_chunker(self, file_type, chunker):
        """Register a chunker for a specific file type.
        
        Args:
            file_type (str): The file type to register the chunker for.
            chunker: The chunker instance to register.
        """
        self.chunkers[file_type] = chunker
        logging.info(f"Registered chunker for file type: {file_type}")
    
    def get_chunker(self, file_type):
        """Get a chunker for a specific file type.
        
        Args:
            file_type (str): The file type to get a chunker for.
            
        Returns:
            The chunker instance for the specified file type,
            or None if no chunker is registered for the file type.
        """
        if file_type in self.chunkers:
            return self.chunkers[file_type]
        else:
            logging.warning(f"No chunker registered for file type: {file_type}")
            return None
    
    def determine_file_type(self, file_path):
        """Determine the file type based on extension.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            str: File type (e.g., 'cpp', 'python').
        """
        import os
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.cpp', '.hpp', '.h', '.cc']:
            return 'cpp'
        elif ext == '.py':
            return 'python'
        else:
            return 'generic'  # Use generic for unknown file types
    
    def chunk_file(self, file_path):
        """Chunk a file using the appropriate chunker.
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            list: List of string chunks.
        """
        file_type = self.determine_file_type(file_path)
        chunker = self.get_chunker(file_type)
        
        if chunker:
            return chunker.chunk_file(file_path)
        else:
            logging.warning(f"No chunker available for file type: {file_type}")
            # Use the generic TokenBasedChunker as fallback
            if 'generic' in self.chunkers:
                logging.info(f"Using generic chunker for file type: {file_type}")
                return self.chunkers['generic'].chunk_file(file_path)
            else:
                logging.error(f"No generic chunker available")
                return [] 