"""
Document Ingestion Engine

This Module provides a comprehensive document ingestion system that can process
multiple file formts(PDF, DOCX, TXT, etc.), extract clean text with metedata, 
and prepare documents for RAG processing with optimized chunking strategies.

Key Components:
- Document and DocumentChunk data structures
= BaseDocumentProcessor interface for extensibility
- Specific processors for different file formats
- DocumentIngestionEngine for orchestrating the ingestion process
- Comprehensive error handling and logging
- Metadata extraction and chunking strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import hashlib
import mimetypes
import os
from datetime import datetime
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """
    Inddividual document chunk for RAG processing.
    
    Args:
        text: The actual text content of the chunk.
        chunk_id: Unique identifier for the chunk.
        start_idx: Starting chracter index in the original document.
        end_idx: Ending chracter index in the original document.
        metadata: Additional metadata associated with the chunk.
        document_id: Identifier of the parent document.
    """
    text: str
    chunk_id: str
    start_idx: int
    end_idx: int
    metadata: Dict[str,Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate chunk creation."""
        if not self.text.strip():
            raise ValueError("Chunk text cannot be empty.")
        if self.end_idx <= self.start_idx:
            raise ValueError("End index must be greater than start index.")
    
    @property
    def word_count(self) -> int:
        """Get the total word count for this chunk."""
        return len(self.text.split())
    
    @property
    def char_count(self)->int:
        """Get the total character count for this chunk."""
        return len(self.text)

@dataclass
class Document:
    """
    Standardized document representation for RAG processing.
    
    Args:
        content: Full text content of the document.
        metadata: Document metadata including title, author, creation date, etc.
        source_path: Original file path or URL
        document_id: Unique identifier for the document.
        chunks: List of document chunks.
        processting_stats: Statistics about the processing.
    """

    content: str
    metadata: Dict[str, Any]
    source_path: str
    document_id: str
    chunks:Optional[List[DocumentChunk]] = None
    processing_stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize document with computed metadata."""
        if not self.document_id:
            self.document_id = self._generate_document_id()
        
        self.metadata.update({
            'word_count': len(self.content.split()),
            'char_count': len(self.content),
            'created_at': datetime.now().isoformat()
        })
    
    def _generate_document_id(self) -> str:
        """Generate a unique document ID based on content hash."""
        content_hash = hashlib.md5(
            f"{self.source_path}{self.content[:1000]}".encode()
        ).hexdigest()
        return f"doc_{content_hash[:16]}"
    
    @property
    def word_count(self) -> int:
        """Get the total word count for the document."""
        return self.metadata.get('word_count', 0)

    @property
    def chunk_count(self) -> int:
        """Get the total number of chunks in the document."""
        return len(self.chunks) if self.chunks else 0

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to the document."""
        self.chunks = chunks
        for chunk in chunks:
            chunk.document_id = self.document_id

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a chunk by its ID."""
        if not self.chunks:
            return None
        return next((chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None)


@dataclass
class ProcessingStats:
    """
    Track processing performance and statistics.
    """
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
    avg_file_size: float = 0.0
    error :List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of processed files."""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100
    
    def add_error(self,error:str) ->None:
        """Add an error message to the stats."""
        self.error.append(f"{datetime.now().isoformat()}: {error}")
        logger.error(f"Error: {error}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to a dictionary for easy serialization."""
        return {
            'total_files': self.total_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'success_rate': round(self.success_rate,2),
            'total_chunks': self.total_chunks,
            'processing_time': round(self.processing_time, 2),
            'avg_file_size': round(self.avg_file_size, 2),
            'error_count': len(self.error),
        }
    
# Custom Exceptions for Document Processing
class DocumentIngestionError(Exception):
    """Base exception for document ingestion errors."""
    pass


class UnsupportedFormatError(DocumentIngestionError):
    """Raised when file format is not supported."""
    pass


class FileAccessError(DocumentIngestionError):
    """Raised when file cannot be accessed or read."""
    pass


class ProcessingError(DocumentIngestionError):
    """Raised when document processing fails."""
    pass


class ValidationError(DocumentIngestionError):
    """Raised when document validation fails."""
    pass

class BaseDocumentProcessor(ABC):
    """
    Abstract base class for document processors.
    Provides a common interface for processing different document formats.
    (PDF, DOCX, TXT, etc.)
    """

    def __init__(self, config: Optional[Dict[str,Any]] = None):
        """
        Initialize the processor with optional configuration.
        
        Args:
            config: Optional configuration dictionary for processor settings.
        """
        self.config = config or {}
        self.supported_extensions = self._get_supported_extensions()

    @abstractmethod
    def _get_supported_extensions(self) -> List[str]:
        """
        Get the list of supported file extensions for this processor.
        """
        pass

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from the given file.
        
        Args:
            file_path: Path to the document file.
        
        Returns:
            Extracted text content.
        
        Raises:
            FileAccessError: If the file cannot be accessed or read.
            ProcessingError: If text extraction fails.
        """
        pass

    @abstractmethod
    def extract_metadata(self, file_path:str) -> Dict[str,Any]:
        """
        Extract metadata from the given file.
        
        Args:
            file_path: Path to the document file.
        
        Returns:
            Dictionary containing extracted metadata.
        """
        pass

    def validate_file(self, file_path:str) -> None:
        """
        Validate the file before processing.
        
        Args:
            file_path: Path to the document file.
        
        Raises:
            FileAccessError: If the file cannot be accessed or read.
            UnsupportedFormatError: If the file format is not supported.
        """

        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileAccessError(f"File not found: {file_path}")
        if not path_obj.is_file():
            raise FileAccessError(f"Path is not a file: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise FileAccessError(f"File is not readable: {file_path}")
        
        #Check file extension
        file_extension = path_obj.suffix.lower()
        if file_extension not in self.supported_extensions:
            raise UnsupportedFormatError(
                f"Unsupported file format: {file_extension}. Supported formats: {self.supported_extensions}"
            )
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get file metadata including size and MIME type.
        
        Args:
            file_path: Path to the document file.
        
        Returns:
            Dictionary containing file metadata.
        """
        path_obj = Path(file_path)
        stat = path_obj.stat()

        mime_type, _ = mimetypes.guess_type(file_path)

        return {
            'file_name': path_obj.name,
            'file_extension': path_obj.suffix.lower(),
            'file_size': stat.st_size,
            'mime_type': mime_type,
            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'file_hash': self._calculate_file_hash(file_path)
        }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of the file for deduplication.
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return ""
    
    def process(self, file_path: str) -> Document:
        """
        Process the document file and return a standardized Document object.

        This is the main entry point for document processing.
        
        Args:
            file_path: Path to the document file.
        
        Returns:
            Document object containing extracted content and metadata.
        
        Raises:
            DocumentIngestionError: If any processing error occurs.
        """
        
        
        try:
            self.validate_file(file_path)
            text = self.extract_text(file_path)
            file_metadata = self.get_file_metadata(file_path)
            doc_metadata = self.extract_metadata(file_path)
            combined_metadata = {**file_metadata, **doc_metadata}
            
            document = Document(
                content=text,
                metadata=combined_metadata,
                source_path=file_path,
                document_id=""
            )
            logger.info(f"Successfully processed document: {file_path}")
            return document
        except Exception as e:
            raise ProcessingError(f"Failed to process document {file_path}: {e}")
    
class DocumentValidator:
    """
    Validate documents and chunks to ensure quality and consistency.
    """

    def __init__(self, config: Optional[Dict[str,Any]] = None):
        self.config = config or {}
        self.min_content_length = self.config.get('min_content_length', 10)
        self.max_content_length = self.config.get('max_content_length', 10_000_000)
        self.min_chunk_length = self.config.get('min_chunk_length', 50)

    def validate_document(self, document: Document) -> bool:
        """
        Validate the entire document.
        
        Args:
            document: Document object to validate.
        
        Returns:
            True if document is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if len(document.content) < self.min_content_length:
            raise ValidationError(
                f"Document content is too short: {len(document.content)} characters."
                f"(minimum: {self.min_content_length} characters)"
            )
        
        if len(document.content) > self.max_content_length:
            raise ValidationError(
                f"Document content is too long: {len(document.content)} characters."
                f"(maximum: {self.max_content_length} characters)"
            )
        
        required_fields = ['document_id', 'source_path']
        for field in required_fields:
            if not getattr(document, field,None):
                raise ValidationError(f"Missing required metadata field: {field}")

        if not isinstance(document.metadata, dict):
            raise ValidationError("Document metadata must be a dictionary.")
        
        return True

    def validate_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Validate chunk quality and structure.

        Args:
            chunk: List of DocumentChunk objects to validate.
        Returns:
            True if all chunks are valid, False otherwise.
        Raises:
            ValidationError: If any chunk validation fails.
        """

        if not chunks:
            raise ValidationError("No chunks provided for validation.")
        
        for i,chunk in enumerate(chunks):
            if(len(chunk.text) < self.min_chunk_length):
                raise ValidationError(
                    f"Chunk {i} is too short: {len(chunk.text)} characters."
                    f"(minimum: {self.min_chunk_length} characters)"
                )
            
            if not chunk.chunk_id:
                raise ValidationError(f"Chunk {i} is missing chunk_id.")
            
            if not chunk.start_idx < chunk.end_idx:
                raise ValidationError(
                    f"Chunk {i} has invalid indices: start_idx={chunk.start_idx}, end_idx={chunk.end_idx}."
                )
            
        return True

        