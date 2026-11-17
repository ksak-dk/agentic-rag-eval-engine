"""
app/retrieval/chunker.py
========================
Structured document chunker with two operating modes:

  FIXED    — Sliding-window character chunking with configurable overlap.
             Mirrors PIPELINE_VERSION=v1 semantics; baseline for benchmarking.

  SEMANTIC — Markdown header-aware (H1 / H2 / H3) structured chunking.
             Splits on structural boundaries first, then enforces a per-chunk
             max-token budget by further splitting on paragraph / sentence
             boundaries. Tracks the full header ancestry in ChunkMetadata.
             Used by PIPELINE_VERSION=v2 and v3.

All public types are Pydantic V2 models with frozen=True — safe to hash,
cache, and ship directly as Qdrant payload dicts.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Conservative chars-per-token ratio for multilingual text.
# Avoids a mandatory tokenizer dependency at chunk time.
_CHARS_PER_TOKEN: float = 3.5

# ATX Markdown headers H1-H3 at line start (captures hashes and title text).
_HEADER_RE: re.Pattern[str] = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Two-or-more consecutive newlines = paragraph boundary.
_PARAGRAPH_SEP_RE: re.Pattern[str] = re.compile(r"\n{2,}")

# Sentence boundary: period / ! / ? followed by whitespace.
_SENTENCE_END_RE: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ChunkMode(StrEnum):
    """Chunking strategy selector passed through ChunkerConfig."""

    FIXED = "fixed"  # v1 baseline: character window + overlap
    SEMANTIC = "semantic"  # v2/v3: Markdown-header + paragraph boundary


# ---------------------------------------------------------------------------
# Pydantic V2 Data Models
# ---------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Strongly-typed metadata for a single DocumentChunk.

    All fields are JSON-serialisable primitives so they map 1-to-1 onto
    Qdrant payload fields without further transformation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    doc_id: str = Field(
        description=(
            "Stable UUID v4 identifier for the source document. "
            "Every chunk from the same document shares this id."
        ),
    )
    source: str = Field(
        description="File path or URL that identifies the source document.",
    )
    section_path: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of ancestor Markdown header texts leading to this chunk. "
            "E.g. ['Architecture', 'Retrieval', 'Hybrid Search']. "
            "Empty list for FIXED mode or preamble text before the first header."
        ),
    )
    chunk_index: int = Field(
        ge=0,
        description="Zero-based position of this chunk within the document.",
    )

    @field_validator("doc_id")
    @classmethod
    def _doc_id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("doc_id must be a non-empty string.")
        return v

    @field_validator("source")
    @classmethod
    def _source_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source must be a non-empty string.")
        return v


class DocumentChunk(BaseModel):
    """A single text segment ready for embedding and Qdrant upsert."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Stripped text content of the chunk.")
    metadata: ChunkMetadata
    char_count: int = Field(ge=0, description="Character length of text.")
    approx_tokens: int = Field(ge=1, description="Estimated token count (char_count / 3.5).")

    @classmethod
    def build(cls, *, text: str, metadata: ChunkMetadata) -> DocumentChunk:
        """Factory that strips text and computes derived fields."""
        text = text.strip()
        return cls(
            text=text,
            metadata=metadata,
            char_count=len(text),
            approx_tokens=max(1, round(len(text) / _CHARS_PER_TOKEN)),
        )


class ChunkerConfig(BaseModel):
    """Runtime parameters for DocumentChunker.

    Instantiate once and reuse — the model is frozen (immutable).
    """

    model_config = ConfigDict(frozen=True)

    mode: ChunkMode = Field(
        default=ChunkMode.SEMANTIC,
        description="Chunking strategy. FIXED for v1 baseline, SEMANTIC for v2/v3.",
    )

    # --- FIXED mode ---
    fixed_chunk_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Target chunk size in characters for FIXED mode.",
    )
    fixed_overlap: int = Field(
        default=64,
        ge=0,
        description="Character overlap between consecutive FIXED chunks.",
    )

    # --- SEMANTIC mode ---
    max_tokens_per_chunk: int = Field(
        default=400,
        ge=64,
        le=2048,
        description=(
            "Maximum estimated tokens per semantic chunk. "
            "Sections exceeding this budget are further split on paragraph boundaries."
        ),
    )
    min_chunk_chars: int = Field(
        default=40,
        ge=1,
        description=(
            "Minimum character length for a standalone chunk. "
            "Shorter trailing segments are merged into the preceding chunk."
        ),
    )

    @model_validator(mode="after")
    def _overlap_lt_chunk_size(self) -> ChunkerConfig:
        if self.fixed_overlap >= self.fixed_chunk_size:
            raise ValueError(
                f"fixed_overlap ({self.fixed_overlap}) must be strictly less than "
                f"fixed_chunk_size ({self.fixed_chunk_size})."
            )
        return self


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _split_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window character splitter with overlap.

    Args:
        text:       Pre-stripped source text.
        chunk_size: Maximum characters per window.
        overlap:    Characters repeated at the start of the next window.

    Returns:
        List of non-empty, stripped string segments.
    """
    if not text:
        return []
    segments: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        if end >= len(text):
            break
        start += step
    return segments


def _split_oversized(text: str, max_chars: int, min_chars: int) -> list[str]:
    """Split a text block that exceeds *max_chars* on semantic boundaries.

    Priority order:
      1. Paragraph boundaries (blank lines)
      2. Sentence boundaries (period / ! / ? + whitespace)
      3. Hard character cut (last resort — should be rare)

    Short trailing pieces (< min_chars) are merged into the previous segment.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    # --- paragraph-level pass ---
    paragraphs = [p.strip() for p in _PARAGRAPH_SEP_RE.split(text) if p.strip()]

    result: list[str] = []
    buffer_parts: list[str] = []
    buffer_len = 0

    for para in paragraphs:
        para_len = len(para)

        # Paragraph itself is oversized → sentence-level pass
        if para_len > max_chars:
            if buffer_parts:
                result.append("\n\n".join(buffer_parts))
                buffer_parts = []
                buffer_len = 0
            sentences = [s.strip() for s in _SENTENCE_END_RE.split(para) if s.strip()]
            sent_buf: list[str] = []
            sent_len = 0
            for sent in sentences:
                slen = len(sent)
                if slen > max_chars:
                    # Hard cut — extremely long single sentence
                    if sent_buf:
                        result.append(" ".join(sent_buf))
                        sent_buf = []
                        sent_len = 0
                    # Emit in max_chars slices
                    for i in range(0, slen, max_chars):
                        slice_text = sent[i : i + max_chars].strip()
                        if slice_text:
                            result.append(slice_text)
                elif sent_len + slen + 1 > max_chars and sent_buf:
                    result.append(" ".join(sent_buf))
                    sent_buf = [sent]
                    sent_len = slen
                else:
                    sent_buf.append(sent)
                    sent_len += slen + 1
            if sent_buf:
                result.append(" ".join(sent_buf))
            continue

        # Normal paragraph fits — check if adding it overflows the buffer
        if buffer_len + para_len > max_chars and buffer_parts:
            result.append("\n\n".join(buffer_parts))
            buffer_parts = []
            buffer_len = 0

        buffer_parts.append(para)
        buffer_len += para_len

    if buffer_parts:
        result.append("\n\n".join(buffer_parts))

    # --- merge micro-chunks ---
    merged: list[str] = []
    for chunk in result:
        chunk = chunk.strip()
        if not chunk:
            continue
        if merged and len(chunk) < min_chars:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    return merged


def _parse_markdown_sections(text: str) -> list[tuple[list[str], str]]:
    """Decompose *text* into (section_path, content) pairs using H1-H3 headers.

    Rules:
    - Text before the first header → section_path=[]
    - Each header owns the text until the next header of equal or higher rank.
    - section_path is the full ancestor chain, e.g. ['Intro', 'Background'].
    - Sections with empty content after stripping are silently dropped.

    Returns:
        Ordered list of (section_path, content_text) tuples.
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        # Plain text — no Markdown structure detected.
        stripped = text.strip()
        return [([], stripped)] if stripped else []

    sections: list[tuple[list[str], str]] = []

    # Preamble: text before the first header.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(([], preamble))

    # Active ancestor stack: list of (level, title) pairs.
    # Invariant: always sorted by increasing level.
    header_stack: list[tuple[int, str]] = []

    for i, match in enumerate(matches):
        level = len(match.group(1))  # 1 = H1, 2 = H2, 3 = H3
        title = match.group(2).strip()

        # Trim the stack to only ancestors strictly shallower than current level.
        header_stack = [(lvl, ttl) for lvl, ttl in header_stack if lvl < level]
        header_stack.append((level, title))

        # Content spans from end of this header line to start of the next.
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        if not content:
            continue  # Header with no body — skip; sub-headers will carry the path.

        section_path = [ttl for _, ttl in header_stack]
        sections.append((section_path, content))

    return sections


# ---------------------------------------------------------------------------
# Public Chunker
# ---------------------------------------------------------------------------


class DocumentChunker:
    """Stateless document chunker.

    Thread-safe after construction; ``config`` is frozen and ``chunk()``
    holds no mutable state.

    Example::

        chunker = DocumentChunker(ChunkerConfig(mode=ChunkMode.SEMANTIC))
        chunks: list[DocumentChunk] = chunker.chunk(
            text=markdown_text,
            source="docs/architecture.md",
        )
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config: ChunkerConfig = config or ChunkerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        source: str,
        doc_id: str | None = None,
    ) -> list[DocumentChunk]:
        """Chunk *text* and return an ordered list of DocumentChunk objects.

        Args:
            text:    Raw document text.  Markdown formatting is honoured in
                     SEMANTIC mode and ignored in FIXED mode.
            source:  Logical identifier for the document (file path or URL).
                     Stored verbatim in ChunkMetadata.source.
            doc_id:  Stable document identifier shared by all chunks of the
                     same document.  A UUID v4 is generated when omitted.

        Returns:
            List of DocumentChunk objects in document order.
            Returns an empty list when *text* is blank.
        """
        resolved_id = doc_id or str(uuid.uuid4())
        stripped = text.strip()
        if not stripped:
            return []

        if self.config.mode == ChunkMode.FIXED:
            return self._chunk_fixed(stripped, source, resolved_id)
        return self._chunk_semantic(stripped, source, resolved_id)

    # ------------------------------------------------------------------
    # Fixed-length mode (v1 baseline)
    # ------------------------------------------------------------------

    def _chunk_fixed(self, text: str, source: str, doc_id: str) -> list[DocumentChunk]:
        raw = _split_fixed(
            text,
            chunk_size=self.config.fixed_chunk_size,
            overlap=self.config.fixed_overlap,
        )
        return [
            DocumentChunk.build(
                text=segment,
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    source=source,
                    section_path=[],  # Fixed mode carries no structural context.
                    chunk_index=idx,
                ),
            )
            for idx, segment in enumerate(raw)
        ]

    # ------------------------------------------------------------------
    # Semantic mode (v2 / v3)
    # ------------------------------------------------------------------

    def _chunk_semantic(self, text: str, source: str, doc_id: str) -> list[DocumentChunk]:
        """H1/H2/H3-aware chunker with per-section token budget enforcement."""
        max_chars = round(self.config.max_tokens_per_chunk * _CHARS_PER_TOKEN)

        sections = _parse_markdown_sections(text)

        # Expand oversized sections into smaller sub-chunks, preserving path.
        raw_segments: list[tuple[list[str], str]] = []
        for section_path, content in sections:
            sub_chunks = _split_oversized(content, max_chars, self.config.min_chunk_chars)
            for sub in sub_chunks:
                raw_segments.append((section_path, sub))

        return [
            DocumentChunk.build(
                text=segment,
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    source=source,
                    section_path=section_path,
                    chunk_index=idx,
                ),
            )
            for idx, (section_path, segment) in enumerate(raw_segments)
        ]


# ---------------------------------------------------------------------------
# Convenience factory — maps PIPELINE_VERSION to a pre-configured chunker
# ---------------------------------------------------------------------------


def make_chunker(pipeline_version: str = "v3") -> DocumentChunker:
    """Return a DocumentChunker pre-configured for the given pipeline version.

    +---------+---------------+-------------------------------------------+
    | Version | Mode          | Parameters                                |
    +=========+===============+===========================================+
    | v1      | FIXED         | 512 chars, 64 overlap                     |
    | v2      | SEMANTIC      | 400 max_tokens, 40 min_chars              |
    | v3      | SEMANTIC      | identical to v2; agentic loop is external |
    +---------+---------------+-------------------------------------------+
    """
    if pipeline_version == "v1":
        return DocumentChunker(
            ChunkerConfig(
                mode=ChunkMode.FIXED,
                fixed_chunk_size=512,
                fixed_overlap=64,
            )
        )
    # v2 and v3 share the same chunker configuration.
    return DocumentChunker(
        ChunkerConfig(
            mode=ChunkMode.SEMANTIC,
            max_tokens_per_chunk=400,
            min_chunk_chars=40,
        )
    )


MarkdownHeaderChunker = DocumentChunker


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__: Sequence[str] = [
    "ChunkMetadata",
    "ChunkMode",
    "ChunkerConfig",
    "DocumentChunk",
    "DocumentChunker",
    "MarkdownHeaderChunker",
    "make_chunker",
]
