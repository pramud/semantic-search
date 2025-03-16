# chunker.py
import os
from tree_sitter import Language, Parser
import logging
import tiktoken

# Set logging level to DEBUG for detailed output; adjust to INFO for production
logging.basicConfig(level=logging.DEBUG)

# Base class for chunkers
class BaseChunker:
    def __init__(self, max_tokens=1500):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text):
        """Count the number of tokens in a text using tiktoken."""
        return len(self.tokenizer.encode(text))
        
    def chunk_file(self, file_path):
        raise NotImplementedError("Subclasses must implement chunk_file")

# Token-based chunker for any file type
class TokenBasedChunker(BaseChunker):
    def __init__(self, max_tokens=1500, overlap_tokens=100):
        super().__init__(max_tokens)
        self.overlap_tokens = overlap_tokens
        
    def chunk_file(self, file_path):
        """Chunk a file based on token count."""
        logging.info(f"Starting to chunk file with token-based approach: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                source_code = f.read()
                
            # Count tokens in the entire file
            total_tokens = self.count_tokens(source_code)
            logging.info(f"File {file_path} has {total_tokens} tokens")
            
            # If file is small enough, return it as a single chunk
            if total_tokens <= self.max_tokens:
                logging.info(f"File {file_path} fits in a single chunk")
                header = f"File: {file_path}\nFunction: entire_file\nLines: 1-{source_code.count(os.linesep) + 1}\n"
                return [header + source_code]
                
            # Otherwise, chunk the file
            return self.create_overlapping_chunks(source_code, file_path)
            
        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")
            return []
            
    def create_overlapping_chunks(self, text, file_path):
        """Create overlapping chunks based on token count."""
        chunks = []
        lines = text.split(os.linesep)
        
        start_line = 0
        current_chunk = []
        current_tokens = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)
            
            # If adding this line would exceed max_tokens, create a chunk
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunk_text = os.linesep.join(current_chunk)
                end_line = start_line + len(current_chunk) - 1
                header = f"File: {file_path}\nFunction: chunk_{len(chunks)+1}\nLines: {start_line+1}-{end_line+1}\n"
                chunks.append(header + chunk_text)
                
                # Calculate overlap
                overlap_size = 0
                overlap_tokens = 0
                overlap_lines = []
                
                # Add lines from the end of the current chunk for overlap
                for line in reversed(current_chunk):
                    line_token_count = self.count_tokens(line)
                    if overlap_tokens + line_token_count <= self.overlap_tokens:
                        overlap_lines.insert(0, line)
                        overlap_tokens += line_token_count
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_lines
                current_tokens = overlap_tokens
                start_line = end_line - len(overlap_lines) + 1
            
            # Add the current line to the chunk
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = os.linesep.join(current_chunk)
            end_line = start_line + len(current_chunk) - 1
            header = f"File: {file_path}\nFunction: chunk_{len(chunks)+1}\nLines: {start_line+1}-{end_line+1}\n"
            chunks.append(header + chunk_text)
        
        logging.info(f"Created {len(chunks)} token-based chunks for {file_path}")
        return chunks

# C++-specific chunker with integrated logic
class CppChunker(BaseChunker):
    def __init__(self, languages_lib_path, max_tokens=1500, overlap_tokens=100):
        super().__init__(max_tokens)
        logging.info(f"Initializing CppChunker with languages_lib_path: {languages_lib_path}")
        self.language = Language(languages_lib_path, 'cpp')
        self.parser = Parser()
        self.parser.set_language(self.language)
        self.overlap_tokens = overlap_tokens

    def chunk_file(self, file_path):
        """Chunk a C++ file into semantic units based on token count."""
        logging.info(f"Starting to chunk file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                source_code = f.read()
                
            # Count tokens in the entire file
            total_tokens = self.count_tokens(source_code)
            logging.info(f"File {file_path} has {total_tokens} tokens")
            
            # If file is small enough, return it as a single chunk
            if total_tokens <= self.max_tokens:
                logging.info(f"File {file_path} fits in a single chunk")
                header = f"File: {file_path}\nFunction: entire_file\nLines: 1-{source_code.count(os.linesep) + 1}\n"
                return [header + source_code]
                
            # Otherwise, parse and chunk by functions
            source_bytes = source_code.encode('utf8')
            tree = self.parser.parse(source_bytes)
            logging.debug(f"Parsed AST for {file_path}")
            chunks = self.extract_chunks_from_node(tree.root_node, source_bytes, file_path)
            logging.info(f"Successfully chunked {file_path} into {len(chunks)} chunks")
            return chunks
        except (IOError, OSError) as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            return []
        except Exception as e:
            logging.error(f"Failed to parse or chunk file {file_path}: {e}")
            return []

    def extract_chunks_from_node(self, node, source_bytes, file_name):
        """Recursively extract semantic chunks from an AST node with token count awareness."""
        chunks = []
        if node.type == 'function_definition':
            logging.debug(f"Processing function_definition node at lines {node.start_point[0]+1}-{node.end_point[0]+1}")
            func_name = "unknown"
            for child in node.children:
                if child.type == 'function_declarator':
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            func_name = source_bytes[subchild.start_byte:subchild.end_byte].decode("utf8")
                            logging.debug(f"Identified function: {func_name}")
                            break
                    break
            start_byte = node.start_byte
            prev = node.prev_sibling
            while prev and prev.type == 'comment':
                start_byte = prev.start_byte
                prev = prev.prev_sibling
            entire_function_text = source_bytes[start_byte:node.end_byte].decode("utf8")
            
            # Count tokens in the function
            function_tokens = self.count_tokens(entire_function_text)
            logging.debug(f"Function {func_name} has {function_tokens} tokens")
            
            header = f"File: {file_name}\nFunction: {func_name}\nLines: {node.start_point[0]+1}-{node.end_point[0]+1}\n"
            
            # If function fits within token limit, add it as a single chunk
            if function_tokens <= self.max_tokens:
                chunks.append(header + entire_function_text)
                logging.debug(f"Added full function chunk for {func_name}")
            else:
                # Function is too large, split it into overlapping chunks
                logging.debug(f"Function {func_name} exceeds token limit, splitting into chunks")
                function_chunks = self.split_by_tokens(entire_function_text, self.max_tokens, self.overlap_tokens)
                for i, chunk_text in enumerate(function_chunks):
                    chunk_header = f"File: {file_name}\nFunction: {func_name} (part {i+1}/{len(function_chunks)})\nLines: {node.start_point[0]+1}-{node.end_point[0]+1}\n"
                    chunks.append(chunk_header + chunk_text)
                    logging.debug(f"Added function part chunk for {func_name}, part {i+1}/{len(function_chunks)}")
        
        # Process child nodes
        for child in node.children:
            chunks.extend(self.extract_chunks_from_node(child, source_bytes, file_name))
        
        return chunks
        
    def split_by_tokens(self, text, max_tokens, overlap_tokens):
        """Split text into chunks based on token count with overlap.
        
        Args:
            text (str): Text to split.
            max_tokens (int): Maximum tokens per chunk.
            overlap_tokens (int): Tokens to overlap for context.
            
        Returns:
            list: List of string chunks.
        """
        chunks = []
        lines = text.split('\n')
        
        current_chunk_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            # If adding this line would exceed max_tokens and we already have content, create a chunk
            if current_tokens + line_tokens > max_tokens and current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines)
                chunks.append(chunk_text)
                
                # Calculate overlap
                overlap_lines = []
                overlap_tokens_count = 0
                
                # Add lines from the end of the current chunk for overlap
                for overlap_line in reversed(current_chunk_lines):
                    overlap_line_tokens = self.count_tokens(overlap_line)
                    if overlap_tokens_count + overlap_line_tokens <= overlap_tokens:
                        overlap_lines.insert(0, overlap_line)
                        overlap_tokens_count += overlap_line_tokens
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk_lines = overlap_lines
                current_tokens = overlap_tokens_count
            
            # Handle the case where a single line exceeds max_tokens
            if line_tokens > max_tokens:
                if current_chunk_lines:
                    chunk_text = '\n'.join(current_chunk_lines)
                    chunks.append(chunk_text)
                    current_chunk_lines = []
                    current_tokens = 0
                
                # Split the long line into smaller pieces
                words = line.split(' ')
                current_line_part = []
                current_line_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word + ' ')
                    if current_line_tokens + word_tokens > max_tokens and current_line_part:
                        part_text = ' '.join(current_line_part)
                        chunks.append(part_text)
                        
                        # Calculate overlap for line parts
                        overlap_words = []
                        overlap_word_tokens = 0
                        
                        for overlap_word in reversed(current_line_part):
                            overlap_word_token_count = self.count_tokens(overlap_word + ' ')
                            if overlap_word_tokens + overlap_word_token_count <= overlap_tokens:
                                overlap_words.insert(0, overlap_word)
                                overlap_word_tokens += overlap_word_token_count
                            else:
                                break
                        
                        current_line_part = overlap_words
                        current_line_tokens = overlap_word_tokens
                    
                    current_line_part.append(word)
                    current_line_tokens += word_tokens
                
                if current_line_part:
                    part_text = ' '.join(current_line_part)
                    chunks.append(part_text)
                
                current_chunk_lines = []
                current_tokens = 0
                continue
            
            # Add the current line to the chunk
            current_chunk_lines.append(line)
            current_tokens += line_tokens
        
        # Add the last chunk if there's anything left
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append(chunk_text)
        
        return chunks

    # Keeping the old methods for compatibility
    def split_function_body(self, body_node, source_bytes, max_chars):
        """Split a function body into chunks of complete statements.
        
        Args:
            body_node: AST node of the function body (compound_statement).
            source_bytes (bytes): Source code in bytes.
            max_chars (int): Maximum characters per chunk.
        
        Returns:
            list: List of tuples (start_line, end_line, chunk_text).
        """
        statements = [child for child in body_node.children if child.type not in ['{', '}']]
        if not statements:
            return []
        
        chunks = []
        i = 0
        while i < len(statements):
            j = i
            chunk_size = 0
            while j < len(statements):
                stmt = statements[j]
                stmt_text = source_bytes[stmt.start_byte:stmt.end_byte].decode("utf8")
                if chunk_size + len(stmt_text) > max_chars and j > i:
                    break
                chunk_size += len(stmt_text)
                j += 1
            if j == i:  # Single statement exceeds max_chars
                stmt = statements[i]
                stmt_text = source_bytes[stmt.start_byte:stmt.end_byte].decode("utf8")
                sub_chunks = self.further_chunk(stmt_text, max_chars)
                base_line = stmt.start_point[0] + 1
                for sub_chunk in sub_chunks:
                    start_idx = stmt_text.index(sub_chunk)
                    lines_before = stmt_text[:start_idx].count('\n')
                    start_line = base_line + lines_before
                    end_line = start_line + sub_chunk.count('\n')
                    chunks.append((start_line, end_line, sub_chunk))
                i += 1
            else:
                chunk_text = source_bytes[statements[i].start_byte:statements[j-1].end_byte].decode("utf8")
                start_line = statements[i].start_point[0] + 1
                end_line = statements[j-1].end_point[0] + 1
                chunks.append((start_line, end_line, chunk_text))
                i = j
        return chunks

    def further_chunk(self, text, max_chars=1500, overlap=100):
        """Split text into smaller chunks with overlap.
        
        Args:
            text (str): Text to split.
            max_chars (int): Maximum characters per chunk.
            overlap (int): Characters to overlap for context.
        
        Returns:
            list: List of string chunks.
        """
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + max_chars, text_length)
            split_positions = [text.rfind(c, start, end) for c in [';', '}']]
            last_split = max(split_positions) if split_positions else -1
            if last_split != -1 and last_split > start:
                end = last_split + 1
            chunk = text[start:end]
            chunks.append(chunk)
            if end == text_length:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

# Python-specific chunker with function detection
class PythonChunker(BaseChunker):
    def __init__(self, max_tokens=1500, overlap_tokens=100):
        super().__init__(max_tokens)
        self.overlap_tokens = overlap_tokens
        
    def chunk_file(self, file_path):
        """Chunk a Python file into semantic units based on token count."""
        logging.info(f"Starting to chunk Python file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                source_code = f.read()
                
            # Count tokens in the entire file
            total_tokens = self.count_tokens(source_code)
            logging.info(f"Python file {file_path} has {total_tokens} tokens")
            
            # If file is small enough, return it as a single chunk
            if total_tokens <= self.max_tokens:
                logging.info(f"File {file_path} fits in a single chunk")
                header = f"File: {file_path}\nFunction: entire_file\nLines: 1-{source_code.count(os.linesep) + 1}\n"
                return [header + source_code]
                
            # Otherwise, extract functions and classes
            chunks = self.extract_python_functions(source_code, file_path)
            logging.info(f"Successfully chunked Python file {file_path} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logging.error(f"Failed to process Python file {file_path}: {e}")
            return []
    
    def extract_python_functions(self, source_code, file_path):
        """Extract functions and classes from Python code."""
        chunks = []
        lines = source_code.split('\n')
        
        # Simple function/class detection
        current_function = []
        current_function_name = None
        current_function_start = 0
        current_function_tokens = 0
        in_function = False
        indent_level = 0
        
        for i, line in enumerate(lines):
            # Check for function or class definition
            stripped = line.strip()
            if (stripped.startswith('def ') or stripped.startswith('class ')) and not in_function:
                # Start of a new function or class
                if current_function:
                    # Process the previous non-function code as a chunk if it exists
                    chunk_text = '\n'.join(current_function)
                    if self.count_tokens(chunk_text) <= self.max_tokens:
                        header = f"File: {file_path}\nFunction: non_function_code\nLines: {current_function_start+1}-{i}\n"
                        chunks.append(header + chunk_text)
                    else:
                        # Split large non-function code
                        sub_chunks = self.split_by_tokens(chunk_text, self.max_tokens, self.overlap_tokens)
                        for j, sub_chunk in enumerate(sub_chunks):
                            sub_header = f"File: {file_path}\nFunction: non_function_code_part_{j+1}\nLines: {current_function_start+1}-{i}\n"
                            chunks.append(sub_header + sub_chunk)
                
                # Start tracking the new function
                current_function = [line]
                if stripped.startswith('def '):
                    current_function_name = stripped[4:].split('(')[0].strip()
                    logging.debug(f"Found Python function: {current_function_name}")
                else:  # class
                    current_function_name = stripped[6:].split('(')[0].strip()
                    logging.debug(f"Found Python class: {current_function_name}")
                
                current_function_start = i
                current_function_tokens = self.count_tokens(line)
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                
            elif in_function:
                # Check if we're still in the function/class (based on indentation)
                if line.strip() and not line.startswith('#') and len(line) - len(line.lstrip()) <= indent_level:
                    # End of function/class
                    in_function = False
                    
                    # Process the function
                    function_text = '\n'.join(current_function)
                    function_tokens = self.count_tokens(function_text)
                    
                    if function_tokens <= self.max_tokens:
                        # Function fits in a single chunk
                        header = f"File: {file_path}\nFunction: {current_function_name}\nLines: {current_function_start+1}-{i}\n"
                        chunks.append(header + function_text)
                    else:
                        # Split large function
                        sub_chunks = self.split_by_tokens(function_text, self.max_tokens, self.overlap_tokens)
                        for j, sub_chunk in enumerate(sub_chunks):
                            sub_header = f"File: {file_path}\nFunction: {current_function_name} (part {j+1}/{len(sub_chunks)})\nLines: {current_function_start+1}-{i}\n"
                            chunks.append(sub_header + sub_chunk)
                    
                    # Start tracking non-function code
                    current_function = [line]
                    current_function_name = None
                    current_function_start = i
                    current_function_tokens = self.count_tokens(line)
                else:
                    # Still in the function/class
                    current_function.append(line)
                    current_function_tokens += self.count_tokens(line)
            else:
                # Not in a function/class
                current_function.append(line)
                current_function_tokens += self.count_tokens(line)
        
        # Process any remaining code
        if current_function:
            chunk_text = '\n'.join(current_function)
            if in_function:
                # It's a function that reached the end of the file
                if current_function_tokens <= self.max_tokens:
                    header = f"File: {file_path}\nFunction: {current_function_name}\nLines: {current_function_start+1}-{len(lines)}\n"
                    chunks.append(header + chunk_text)
                else:
                    # Split large function
                    sub_chunks = self.split_by_tokens(chunk_text, self.max_tokens, self.overlap_tokens)
                    for j, sub_chunk in enumerate(sub_chunks):
                        sub_header = f"File: {file_path}\nFunction: {current_function_name} (part {j+1}/{len(sub_chunks)})\nLines: {current_function_start+1}-{len(lines)}\n"
                        chunks.append(sub_header + sub_chunk)
            else:
                # It's non-function code
                if current_function_tokens <= self.max_tokens:
                    header = f"File: {file_path}\nFunction: non_function_code\nLines: {current_function_start+1}-{len(lines)}\n"
                    chunks.append(header + chunk_text)
                else:
                    # Split large non-function code
                    sub_chunks = self.split_by_tokens(chunk_text, self.max_tokens, self.overlap_tokens)
                    for j, sub_chunk in enumerate(sub_chunks):
                        sub_header = f"File: {file_path}\nFunction: non_function_code_part_{j+1}\nLines: {current_function_start+1}-{len(lines)}\n"
                        chunks.append(sub_header + sub_chunk)
        
        return chunks
    
    def split_by_tokens(self, text, max_tokens, overlap_tokens):
        """Split text into chunks based on token count with overlap."""
        chunks = []
        lines = text.split('\n')
        
        current_chunk_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            # If adding this line would exceed max_tokens and we already have content, create a chunk
            if current_tokens + line_tokens > max_tokens and current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines)
                chunks.append(chunk_text)
                
                # Calculate overlap
                overlap_lines = []
                overlap_tokens_count = 0
                
                # Add lines from the end of the current chunk for overlap
                for overlap_line in reversed(current_chunk_lines):
                    overlap_line_tokens = self.count_tokens(overlap_line)
                    if overlap_tokens_count + overlap_line_tokens <= overlap_tokens:
                        overlap_lines.insert(0, overlap_line)
                        overlap_tokens_count += overlap_line_tokens
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk_lines = overlap_lines
                current_tokens = overlap_tokens_count
            
            # Handle the case where a single line exceeds max_tokens
            if line_tokens > max_tokens:
                if current_chunk_lines:
                    chunk_text = '\n'.join(current_chunk_lines)
                    chunks.append(chunk_text)
                    current_chunk_lines = []
                    current_tokens = 0
                
                # Split the long line into smaller pieces
                words = line.split(' ')
                current_line_part = []
                current_line_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word + ' ')
                    if current_line_tokens + word_tokens > max_tokens and current_line_part:
                        part_text = ' '.join(current_line_part)
                        chunks.append(part_text)
                        
                        # Calculate overlap for line parts
                        overlap_words = []
                        overlap_word_tokens = 0
                        
                        for overlap_word in reversed(current_line_part):
                            overlap_word_token_count = self.count_tokens(overlap_word + ' ')
                            if overlap_word_tokens + overlap_word_token_count <= overlap_tokens:
                                overlap_words.insert(0, overlap_word)
                                overlap_word_tokens += overlap_word_token_count
                            else:
                                break
                        
                        current_line_part = overlap_words
                        current_line_tokens = overlap_word_tokens
                    
                    current_line_part.append(word)
                    current_line_tokens += word_tokens
                
                if current_line_part:
                    part_text = ' '.join(current_line_part)
                    chunks.append(part_text)
                
                current_chunk_lines = []
                current_tokens = 0
                continue
            
            # Add the current line to the chunk
            current_chunk_lines.append(line)
            current_tokens += line_tokens
        
        # Add the last chunk if there's anything left
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append(chunk_text)
        
        return chunks

# Registry for chunker instances
chunker_instances = {}

def determine_file_type(file_path):
    """Determine the file type based on extension.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        str: File type (e.g., 'cpp', 'python').
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.cpp', '.hpp', '.h', '.cc']:
        return 'cpp'
    elif ext == '.py':
        return 'python'
    else:
        return 'generic'  # Use generic for unknown file types

def get_chunker(file_type, languages_lib_path, max_tokens=1500, overlap_tokens=100):
    """Get or create a chunker instance for the file type.
    
    Args:
        file_type (str): Type of the file (e.g., 'cpp').
        languages_lib_path (str): Path to the tree-sitter languages library.
        max_tokens (int): Maximum tokens per chunk.
        overlap_tokens (int): Tokens to overlap for context.
    
    Returns:
        BaseChunker: Chunker instance for the file type.
    """
    chunker_key = f"{file_type}_{max_tokens}_{overlap_tokens}"
    if chunker_key not in chunker_instances:
        if file_type == 'cpp':
            chunker_instances[chunker_key] = CppChunker(languages_lib_path, max_tokens, overlap_tokens)
        elif file_type == 'python':
            chunker_instances[chunker_key] = PythonChunker(max_tokens, overlap_tokens)
        else:
            # Use TokenBasedChunker for generic file types
            chunker_instances[chunker_key] = TokenBasedChunker(max_tokens, overlap_tokens)
    return chunker_instances[chunker_key]

def chunk_file(file_path, languages_lib_path, max_tokens=1500, overlap_tokens=100, file_type=None):
    """Generic API method to chunk a file.
    
    Args:
        file_path (str): Path to the file.
        languages_lib_path (str): Path to the tree-sitter languages library.
        max_tokens (int): Maximum tokens per chunk.
        overlap_tokens (int): Tokens to overlap for context.
        file_type (str, optional): File type; inferred if None.
    
    Returns:
        list: List of string chunks.
    """
    if file_type is None:
        file_type = determine_file_type(file_path)
    chunker = get_chunker(file_type, languages_lib_path, max_tokens, overlap_tokens)
    return chunker.chunk_file(file_path)

# Example usage
if __name__ == "__main__":
    import tiktoken
    import sys
    
    # Example of token counting
    enc = tiktoken.get_encoding("cl100k_base")
    sample_text = "This is a sample text to count tokens."
    token_count = len(enc.encode(sample_text))
    print(f"Sample text token count: {token_count}")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "./bitcoin/src/core_read.cpp"  # Default example file
        
    if len(sys.argv) > 2:
        max_tokens = int(sys.argv[2])
    else:
        max_tokens = 1500  # Default token limit
    
    # Example of chunking a file
    languages_lib_path = "build/languages.dll"  # Adjust this path as needed
    
    try:
        print(f"Chunking file: {file_path} with max_tokens={max_tokens}")
        file_type = determine_file_type(file_path)
        print(f"Detected file type: {file_type}")
        
        chunks = chunk_file(file_path, languages_lib_path, max_tokens)
        print(f"File was chunked into {len(chunks)} parts")
        
        for idx, chunk in enumerate(chunks, 1):
            # Split the chunk into header and content
            header_lines, content = chunk.split('\n', 3)
            print(f"\n--- Chunk {idx} ---")
            print(header_lines[0])  # File
            print(header_lines[1])  # Function
            print(header_lines[2])  # Lines
            
            # Count tokens in this chunk
            chunk_tokens = len(enc.encode(content))
            print(f"Token count: {chunk_tokens}")
            
            # Print a preview of the content
            content_preview = content[:200] + "..." if len(content) > 200 else content
            print(f"Content preview: {content_preview}")
    except Exception as e:
        print(f"Error: {e}")