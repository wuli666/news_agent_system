"""
News Chunker Service using Chonkie

Provides text chunking for optimal entity extraction.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from chonkie import RecursiveChunker, SemanticChunker, Chunk
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    RecursiveChunker = None
    SemanticChunker = None
    Chunk = None

from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class NewsChunk:
    """Represents a chunk of news text."""
    text: str
    token_count: int
    start_index: int
    end_index: int
    metadata: Dict[str, Any]


class NewsChunker:
    """
    News text chunker using Chonkie library.

    Supports multiple chunking strategies:
    - Recursive: Structure-aware chunking
    - Semantic: Meaning-based chunking
    - Simple: Basic size-based chunking (fallback)
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: str = "recursive"
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy ('recursive', 'semantic', 'simple')
        """
        config = get_config()
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.strategy = strategy

        self._chunker = None
        self._init_chunker()

    def _init_chunker(self):
        """Initialize the appropriate chunker based on strategy."""
        if not CHONKIE_AVAILABLE:
            logger.warning(
                "Chonkie not available, using simple chunking. "
                "Install with: pip install chonkie"
            )
            return

        try:
            if self.strategy == "semantic":
                self._chunker = SemanticChunker(
                    chunk_size=self.chunk_size,
                )
            else:  # recursive (default)
                self._chunker = RecursiveChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            logger.info(f"Initialized {self.strategy} chunker")
        except Exception as e:
            logger.error(f"Failed to initialize chunker: {e}")
            self._chunker = None

    def chunk_text(self, text: str) -> List[NewsChunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Input text to chunk

        Returns:
            List of NewsChunk objects
        """
        if not text or not text.strip():
            return []

        # Use Chonkie if available
        if self._chunker is not None:
            try:
                chunks = self._chunker(text)
                return [
                    NewsChunk(
                        text=chunk.text,
                        token_count=chunk.token_count,
                        start_index=getattr(chunk, 'start_index', 0),
                        end_index=getattr(chunk, 'end_index', len(chunk.text)),
                        metadata={}
                    )
                    for chunk in chunks
                ]
            except Exception as e:
                logger.error(f"Chonkie chunking failed: {e}, falling back to simple")

        # Fallback to simple chunking
        return self._simple_chunk(text)

    def _simple_chunk(self, text: str) -> List[NewsChunk]:
        """
        Simple character-based chunking fallback.

        Args:
            text: Input text

        Returns:
            List of NewsChunk objects
        """
        # Approximate tokens as chars / 4 for Chinese text
        char_size = self.chunk_size * 2  # Roughly 2 chars per token for Chinese
        char_overlap = self.chunk_overlap * 2

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + char_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for delimiter in ["。", "！", "？", ".", "!", "?", "\n"]:
                    last_delim = text[start:end].rfind(delimiter)
                    if last_delim > char_size // 2:
                        end = start + last_delim + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(NewsChunk(
                    text=chunk_text,
                    token_count=len(chunk_text) // 2,  # Approximate
                    start_index=start,
                    end_index=end,
                    metadata={}
                ))

            # Move to next chunk with overlap
            start = end - char_overlap if end < len(text) else end

        return chunks

    def chunk_news_items(
        self,
        news_items: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> List[NewsChunk]:
        """
        Chunk multiple news items.

        Args:
            news_items: List of news item dictionaries
            include_metadata: Whether to include item metadata in chunks

        Returns:
            List of NewsChunk objects
        """
        all_chunks = []

        for i, item in enumerate(news_items):
            # Build text from news item
            text_parts = []

            title = item.get("title", "")
            if title:
                text_parts.append(f"标题：{title}")

            content = item.get("content", item.get("description", ""))
            if content:
                text_parts.append(f"内容：{content}")

            source = item.get("source", item.get("platform", ""))
            if source:
                text_parts.append(f"来源：{source}")

            full_text = "\n".join(text_parts)

            if not full_text.strip():
                continue

            # Chunk the text
            chunks = self.chunk_text(full_text)

            # Add metadata if requested
            if include_metadata:
                for chunk in chunks:
                    chunk.metadata = {
                        "news_index": i,
                        "title": title,
                        "source": source,
                        "url": item.get("url", ""),
                    }

            all_chunks.extend(chunks)

        return all_chunks

    def get_chunked_texts(self, text: str) -> List[str]:
        """
        Simple helper to get just the text content of chunks.

        Args:
            text: Input text

        Returns:
            List of chunk text strings
        """
        chunks = self.chunk_text(text)
        return [chunk.text for chunk in chunks]


def create_chunker(
    strategy: str = "recursive",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> NewsChunker:
    """
    Factory function to create a chunker.

    Args:
        strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        Configured NewsChunker instance
    """
    return NewsChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy
    )
