"""
Document Chunking Strategies for RAG Pipeline

This module provides various chunking strategies to split documents into
optimal pieces for vector storage and retrieval.

Key Chunking Strategies:
- FixedSizeChunker: Split by character/token count
- SemanticChunker: Split by semantic boundaries  
- ParagraphChunker: Split by paragraphs
- SentenceChunker: Split by sentences
- AdaptiveChunker: Dynamic chunking based on content type
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    text: str
    chunk_id: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Add chunk-specific metadata
        self.metadata.update({
            'chunk_id': self.chunk_id,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'chunk_index': self.chunk_index,
            'char_count': len(self.text),
            'word_count': len(self.text.split())
        })


class BaseChunker(ABC):
    """Base class for document chunking strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_chunk_size = self.config.get('min_chunk_size', 50)
        self.overlap_size = self.config.get('overlap_size', 0)
    
    @abstractmethod
    def chunk_text(self, text: str, doc_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            doc_metadata: Document-level metadata to include in chunks
            
        Returns:
            List of DocumentChunk objects
        """
        pass
    
    def _create_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Create unique chunk identifier."""
        return f"{doc_id}_chunk_{chunk_index:04d}"
    
    def _merge_metadata(self, doc_metadata: Dict[str, Any], chunk_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Merge document and chunk metadata."""
        merged = {}
        if doc_metadata:
            merged.update(doc_metadata)
        merged.update(chunk_metadata)
        return merged
    
    def _filter_small_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove chunks that are too small."""
        return [chunk for chunk in chunks if len(chunk.text.strip()) >= self.min_chunk_size]


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size chunks with optional overlap.
    Good for consistent chunk sizes and processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.overlap_size = self.config.get('overlap_size', 100)
    
    def chunk_text(self, text: str, doc_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Split text into fixed-size chunks."""
        doc_metadata = doc_metadata or {}
        doc_id = doc_metadata.get('doc_id', 'unknown')
        chunks = []
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=self._create_chunk_id(doc_id, chunk_index),
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_index,
                    metadata=self._merge_metadata(doc_metadata, {
                        'chunking_strategy': 'fixed_size',
                        'chunk_size': self.chunk_size,
                        'overlap_size': self.overlap_size
                    })
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.overlap_size, end)
            
            if start >= len(text):
                break
        
        return self._filter_small_chunks(chunks)


class ParagraphChunker(BaseChunker):
    """
    Split text by paragraphs.
    Good for maintaining semantic coherence.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_paragraph_size = self.config.get('max_paragraph_size', 2000)
        self.combine_short_paragraphs = self.config.get('combine_short_paragraphs', True)
    
    def chunk_text(self, text: str, doc_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Split text into paragraph-based chunks."""
        doc_metadata = doc_metadata or {}
        doc_id = doc_metadata.get('doc_id', 'unknown')
        
        # Split by double newlines (paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks = []
        chunk_index = 0
        current_pos = 0
        
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i].strip()
            if not paragraph:
                # Find the position after this empty paragraph
                current_pos = text.find(paragraph, current_pos) + len(paragraph)
                i += 1
                continue
            
            chunk_text = paragraph
            start_pos = text.find(paragraph, current_pos)
            
            # Combine short paragraphs if enabled
            if self.combine_short_paragraphs and len(chunk_text) < self.min_chunk_size:
                combined_paragraphs = [paragraph]
                
                # Look ahead to combine with next paragraphs
                j = i + 1
                while (j < len(paragraphs) and 
                       len('\n\n'.join(combined_paragraphs)) < self.max_paragraph_size):
                    
                    next_para = paragraphs[j].strip()
                    if next_para:
                        combined_paragraphs.append(next_para)
                    j += 1
                
                chunk_text = '\n\n'.join(combined_paragraphs)
                i = j - 1  # Skip the combined paragraphs
            
            # Handle oversized paragraphs
            if len(chunk_text) > self.max_paragraph_size:
                # Split large paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= self.max_paragraph_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunk = self._create_chunk(
                                current_chunk.strip(), doc_id, chunk_index, 
                                start_pos, start_pos + len(current_chunk), doc_metadata
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                            start_pos += len(current_chunk)
                        current_chunk = sentence + " "
                
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk.strip(), doc_id, chunk_index,
                        start_pos, start_pos + len(current_chunk), doc_metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                chunk = self._create_chunk(
                    chunk_text, doc_id, chunk_index,
                    start_pos, start_pos + len(chunk_text), doc_metadata
                )
                chunks.append(chunk)
                chunk_index += 1
            
            current_pos = start_pos + len(chunk_text)
            i += 1
        
        return self._filter_small_chunks(chunks)
    
    def _create_chunk(self, text: str, doc_id: str, chunk_index: int, 
                     start_pos: int, end_pos: int, doc_metadata: Dict[str, Any]) -> DocumentChunk:
        """Helper to create a chunk."""
        return DocumentChunk(
            text=text,
            chunk_id=self._create_chunk_id(doc_id, chunk_index),
            start_char=start_pos,
            end_char=end_pos,
            chunk_index=chunk_index,
            metadata=self._merge_metadata(doc_metadata, {
                'chunking_strategy': 'paragraph',
                'max_paragraph_size': self.max_paragraph_size
            })
        )


class SentenceChunker(BaseChunker):
    """
    Split text by sentences with grouping.
    Good for fine-grained semantic units.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sentences_per_chunk = self.config.get('sentences_per_chunk', 5)
        self.max_chunk_size = self.config.get('max_chunk_size', 1500)
    
    def chunk_text(self, text: str, doc_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Split text into sentence-based chunks."""
        doc_metadata = doc_metadata or {}
        doc_id = doc_metadata.get('doc_id', 'unknown')
        
        # Split into sentences using regex
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text.strip())
        
        chunks = []
        chunk_index = 0
        current_pos = 0
        
        i = 0
        while i < len(sentences):
            chunk_sentences = []
            chunk_text = ""
            
            # Group sentences into chunks
            for j in range(self.sentences_per_chunk):
                if i + j < len(sentences):
                    sentence = sentences[i + j].strip()
                    if sentence:
                        if len(chunk_text + sentence) <= self.max_chunk_size:
                            chunk_sentences.append(sentence)
                            chunk_text += sentence + " "
                        else:
                            break
            
            if chunk_sentences:
                chunk_text = chunk_text.strip()
                start_pos = text.find(chunk_sentences[0], current_pos)
                end_pos = start_pos + len(chunk_text)
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=self._create_chunk_id(doc_id, chunk_index),
                    start_char=start_pos,
                    end_char=end_pos,
                    chunk_index=chunk_index,
                    metadata=self._merge_metadata(doc_metadata, {
                        'chunking_strategy': 'sentence',
                        'sentences_per_chunk': len(chunk_sentences),
                        'max_chunk_size': self.max_chunk_size
                    })
                )
                chunks.append(chunk)
                chunk_index += 1
                current_pos = end_pos
            
            i += len(chunk_sentences) if chunk_sentences else 1
        
        return self._filter_small_chunks(chunks)


class AdaptiveChunker(BaseChunker):
    """
    Adaptive chunking that selects strategy based on content type and characteristics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_strategy = self.config.get('default_strategy', 'paragraph')
        
        # Initialize strategy instances
        self.strategies = {
            'fixed_size': FixedSizeChunker(config),
            'paragraph': ParagraphChunker(config),
            'sentence': SentenceChunker(config)
        }
    
    def chunk_text(self, text: str, doc_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Adaptively choose chunking strategy based on content."""
        doc_metadata = doc_metadata or {}
        
        # Analyze text characteristics
        strategy = self._select_strategy(text, doc_metadata)
        
        # Apply selected strategy
        chunker = self.strategies[strategy]
        chunks = chunker.chunk_text(text, doc_metadata)
        
        # Add adaptive strategy info to metadata
        for chunk in chunks:
            chunk.metadata['adaptive_strategy_used'] = strategy
        
        logger.info(f"Used {strategy} strategy for document, created {len(chunks)} chunks")
        return chunks
    
    def _select_strategy(self, text: str, doc_metadata: Dict[str, Any]) -> str:
        """Select optimal chunking strategy based on content analysis."""
        file_type = doc_metadata.get('file_type', '').lower()
        content_type = doc_metadata.get('content_type', '').lower()
        
        # Strategy selection rules
        if file_type in ['pdf', 'docx'] or 'academic' in content_type:
            # Research papers and formal documents work well with paragraph chunking
            return 'paragraph'
        elif file_type in ['txt', 'md'] and len(text) > 10000:
            # Large text files benefit from fixed-size chunking
            return 'fixed_size'
        elif 'technical' in content_type or 'code' in content_type:
            # Technical content works well with sentence-based chunking
            return 'sentence'
        else:
            return self.default_strategy


class ChunkingPipeline:
    """
    Complete chunking pipeline that processes documents and prepares them for vector storage.
    """
    
    def __init__(self, chunker: BaseChunker, config: Optional[Dict[str, Any]] = None):
        self.chunker = chunker
        self.config = config or {}
        self.validate_chunks = self.config.get('validate_chunks', True)
        self.remove_duplicates = self.config.get('remove_duplicates', True)
    
    def process_document(self, text: str, doc_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Process a document through the complete chunking pipeline.
        
        Args:
            text: Document text to process
            doc_metadata: Document metadata
            
        Returns:
            List of processed and validated chunks
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Generate chunks
        chunks = self.chunker.chunk_text(text, doc_metadata)
        
        # Post-process chunks
        if self.validate_chunks:
            chunks = self._validate_chunks(chunks)
        
        if self.remove_duplicates:
            chunks = self._remove_duplicate_chunks(chunks)
        
        logger.info(f"Processed document into {len(chunks)} chunks")
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text before chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text
    
    def _validate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Validate and filter chunks based on quality criteria."""
        valid_chunks = []
        
        for chunk in chunks:
            # Check minimum length
            if len(chunk.text.strip()) < self.chunker.min_chunk_size:
                continue
            
            # Check if chunk has meaningful content (not just punctuation/whitespace)
            if not re.search(r'[a-zA-Z0-9]', chunk.text):
                continue
            
            valid_chunks.append(chunk)
        
        return valid_chunks
    
    def _remove_duplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove duplicate chunks based on text similarity."""
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            # Simple deduplication based on exact text match
            chunk_text = chunk.text.strip().lower()
            if chunk_text not in seen_texts:
                seen_texts.add(chunk_text)
                unique_chunks.append(chunk)
        
        return unique_chunks


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    sample_text = """
    ChromaDB is a vector database designed for AI applications. It provides persistent storage
    for embeddings and supports semantic search capabilities.
    
    The system uses sentence transformers to generate embeddings from text documents. These
    embeddings are then stored in collections for efficient retrieval.
    
    RAG systems combine retrieval and generation techniques to provide better responses.
    They first retrieve relevant context from a knowledge base, then generate responses
    using that context.
    """
    
    doc_metadata = {
        'doc_id': 'test_doc_001',
        'file_type': 'txt',
        'source': 'documentation'
    }
    
    # Test different chunking strategies
    strategies = {
        'Fixed Size': FixedSizeChunker({'chunk_size': 200, 'overlap_size': 50}),
        'Paragraph': ParagraphChunker(),
        'Sentence': SentenceChunker({'sentences_per_chunk': 3}),
        'Adaptive': AdaptiveChunker()
    }
    
    for name, chunker in strategies.items():
        print(f"\n=== {name} Chunking ===")
        pipeline = ChunkingPipeline(chunker)
        chunks = pipeline.process_document(sample_text, doc_metadata)
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk.text[:100]}...")
            print(f"Metadata: {chunk.metadata}")
            print("-" * 50)