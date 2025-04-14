import hashlib
import filetype
from typing import Tuple, Optional, List, Dict, Any, BinaryIO, Union
from pathlib import Path
import os
import base64
import asyncio
import tempfile
from langchain_core.documents import Document
from loguru import logger
from datetime import datetime

from app.core.metrics import measure_time, DOCUMENT_PROCESSING_DURATION


def generate_hash(data: bytes) -> str:
    """
    Generate a SHA-256 hash of binary data.
    
    Args:
        data: Binary data to hash
        
    Returns:
        Hexadecimal string representation of the hash
    """
    return hashlib.sha256(data).hexdigest()


def get_file_type(file: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine the file type and MIME type of binary data.
    
    Args:
        file: Binary file data
        
    Returns:
        Tuple of (file_extension, mime_type) or (None, None) if type can't be determined
    """
    kind = filetype.guess(file)
    if kind is None:
        # Try to detect text files which filetype may not recognize
        try:
            # Check if it's likely a text file by trying to decode as utf-8
            file[:1024].decode('utf-8')
            return "txt", "text/plain"
        except UnicodeDecodeError:
            return None, None
            
    return kind.extension, kind.mime


def calculate_document_size(document: Document) -> int:
    """
    Calculate the size of a document in bytes.
    
    Args:
        document: Document object
        
    Returns:
        Size in bytes
    """
    text = document.page_content
    return len(text.encode("utf-8"))


def calculate_batch_size(documents: List[Document]) -> int:
    """
    Calculate the total size of a batch of documents in bytes.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    for doc in documents:
        total_size += calculate_document_size(doc)
    return total_size


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get metadata about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file metadata
    """
    path = Path(file_path)
    stats = path.stat()
    
    try:
        with open(path, 'rb') as f:
            first_bytes = f.read(1024)
            file_ext, mime_type = get_file_type(first_bytes)
    except Exception:
        file_ext, mime_type = None, None
        
    if not file_ext:
        file_ext = path.suffix.lstrip('.')
    
    return {
        'name': path.name,
        'path': str(path.absolute()),
        'size': stats.st_size,
        'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
        'file_type': file_ext,
        'mime_type': mime_type,
    }


@measure_time(DOCUMENT_PROCESSING_DURATION, {"file_type": "batch"})
def batch_process_files(
    file_paths: List[Union[str, Path]], 
    processor_func: callable, 
    batch_size: int = 5,
    **kwargs
) -> List[Any]:
    """
    Process multiple files in batches to control memory usage.
    
    Args:
        file_paths: List of paths to process
        processor_func: Function to process each file
        batch_size: Number of files to process in each batch
        **kwargs: Additional arguments to pass to processor_func
        
    Returns:
        Combined results from all processor function calls
    """
    results = []
    
    # Process files in batches
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        batch_results = []
        
        # Process each file in the batch
        for file_path in batch:
            try:
                # Apply file type specific processing
                file_info = get_file_info(file_path)
                file_type = file_info.get('file_type', '')
                
                with measure_time(DOCUMENT_PROCESSING_DURATION, {"file_type": file_type}):
                    result = processor_func(file_path, **kwargs)
                    
                batch_results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                # Continue with other files
        
        # Add batch results to overall results
        if isinstance(batch_results[0], list):
            # Flatten list of lists
            for br in batch_results:
                results.extend(br)
        else:
            results.extend(batch_results)
        
        # Free memory after each batch
        batch_results.clear()
        
    return results


async def async_batch_process_files(
    file_paths: List[Union[str, Path]], 
    processor_func: callable, 
    batch_size: int = 5,
    max_concurrency: int = 3,
    **kwargs
) -> List[Any]:
    """
    Process multiple files in batches asynchronously.
    
    Args:
        file_paths: List of paths to process
        processor_func: Async function to process each file
        batch_size: Number of files to process in each batch
        max_concurrency: Maximum number of concurrent tasks
        **kwargs: Additional arguments to pass to processor_func
        
    Returns:
        Combined results from all processor function calls
    """
    results = []
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_with_semaphore(file_path):
        async with semaphore:
            file_info = get_file_info(file_path)
            file_type = file_info.get('file_type', '')
            
            try:
                with measure_time(DOCUMENT_PROCESSING_DURATION, {"file_type": file_type}):
                    return await processor_func(file_path, **kwargs)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                return []
    
    # Process files in batches
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        
        # Process batch concurrently
        tasks = [process_with_semaphore(file_path) for file_path in batch]
        batch_results = await asyncio.gather(*tasks)
        
        # Add batch results to overall results
        if batch_results and isinstance(batch_results[0], list):
            # Flatten list of lists
            for br in batch_results:
                if br:  # Skip empty results from failed processing
                    results.extend(br)
        else:
            results.extend([r for r in batch_results if r])
        
        # Allow garbage collection
        batch_results = None
        tasks = None
        
    return results


async def save_upload_file(
    file: BinaryIO, 
    destination: Union[str, Path], 
    filename: Optional[str] = None
) -> str:
    """
    Save an uploaded file to disk asynchronously.
    
    Args:
        file: File-like object
        destination: Directory to save the file
        filename: Optional filename (if None, uses the file's name)
        
    Returns:
        Path to the saved file
    """
    # Ensure destination directory exists
    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Get filename
    if not filename:
        filename = Path(getattr(file, 'filename', 'upload.bin')).name
    
    # Generate safe filename to avoid conflicts
    filename_parts = Path(filename).stem, Path(filename).suffix
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    safe_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
    
    # Full path for saving
    file_path = dest_path / safe_filename
    
    # Save file asynchronously
    content = await file.read()
    
    # Run blocking I/O in a thread
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: file_path.write_bytes(content)
    )
    
    logger.info(f"Saved uploaded file to {file_path}")
    return str(file_path)


def encode_file_to_base64(file_path: Union[str, Path]) -> str:
    """
    Encode a file to base64 string.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Base64 encoded string
    """
    path = Path(file_path)
    with open(path, 'rb') as f:
        file_content = f.read()
    
    # Encode to base64
    base64_encoded = base64.b64encode(file_content).decode('utf-8')
    
    # Add MIME type prefix for data URLs if needed
    mime_type = get_file_type(file_content)[1] or 'application/octet-stream'
    return f"data:{mime_type};base64,{base64_encoded}"


def create_temp_copy(file_path: Union[str, Path]) -> str:
    """
    Create a temporary copy of a file to avoid modifying the original.
    
    Args:
        file_path: Path to the original file
        
    Returns:
        Path to the temporary copy
    """
    path = Path(file_path)
    
    # Create temp file with same extension
    suffix = path.suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp_path = temp.name
        
    # Copy content
    with open(path, 'rb') as src, open(temp_path, 'wb') as dst:
        dst.write(src.read())
    
    return temp_path