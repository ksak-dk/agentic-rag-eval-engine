"""
app/cache/semantic_cache.py
===========================
Redis 8 native vector semantic cache backed by RediSearch (HNSW / FLAT).

Design philosophy
-----------------
- Uses ``redis.asyncio`` for non-blocking async I/O.
- Expects 1024-dim dense embeddings (``BAAI/bge-m3``) injected via the
  ``DenseEmbedder`` Protocol from ``app.retrieval.hybrid_store``, keeping unit
  tests 100% GPU-free and Redis-free.
- Enforces cosine similarity threshold (default ``0.92``), configurable TTL,
  and namespace isolation (e.g. ``v1`` / ``v2`` / ``v3``).
- Emits ``cache_hit`` updates to ``RequestMetrics`` via ``contextvars``.
- Fails open on any Redis / network error so the primary RAG pipeline is
  never blocked by cache degradation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import struct
import time
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import Settings, get_settings
from app.core.logging import update_request_metrics
from app.retrieval.hybrid_store import DENSE_DIM, DenseEmbedder

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD: float = 0.92
DEFAULT_TTL_SECONDS: int = 86_400
DEFAULT_REDIS_URL: str = "redis://localhost:6379"
DEFAULT_INDEX_NAME: str = "idx:semantic_cache"
DEFAULT_KEY_PREFIX: str = "rag:cache"
DEFAULT_NAMESPACE: str = "v3"

_TAG_SPECIAL_RE: re.Pattern[str] = re.compile(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/\\\s])")
_RAW_HTML_NOISE_RE: re.Pattern[str] = re.compile(
    r"</?(?:div|p|span|center|img)\b|img\.shields\.io",
    re.IGNORECASE,
)
_FALLBACK_SIGNATURE_PREFIX: str = (
    "I could not find sufficiently relevant and verified context in the knowledge base"
)
_T = TypeVar("_T")


def contains_raw_html_noise(text: str) -> bool:
    """Return True when text contains unstripped layout HTML or Shields.io badge URLs."""
    return bool(_RAW_HTML_NOISE_RE.search(text))


def is_cacheable_response(
    response: str,
    *,
    metadata: dict[str, Any] | None = None,
    is_fallback: bool = False,
) -> bool:
    """Guard semantic cache against fallback templates, blank replies, or raw HTML noise."""
    cleaned = response.strip()
    if not cleaned:
        return False
    if is_fallback:
        return False
    if cleaned.startswith(_FALLBACK_SIGNATURE_PREFIX):
        return False
    if metadata:
        if bool(metadata.get("is_fallback")) or bool(metadata.get("fallback_triggered")):
            return False
        if str(metadata.get("generation_mode", "")).strip().lower() == "fallback":
            return False
    return not contains_raw_html_noise(cleaned)


class VectorIndexAlgorithm(StrEnum):
    """Redis 8 RediSearch vector index algorithms."""

    HNSW = "HNSW"
    FLAT = "FLAT"


# ---------------------------------------------------------------------------
# Pydantic V2 Models
# ---------------------------------------------------------------------------


class CachedResponse(BaseModel):
    """Validated semantic cache hit payload returned by ``SemanticCache.get``."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Original query string stored in cache.")
    response: str = Field(description="Cached LLM response text.")
    namespace: str = Field(default=DEFAULT_NAMESPACE, description="Namespace isolation key.")
    similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between query and cached vector.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional citation, route, or pipeline metadata.",
    )
    created_at: float = Field(
        default=0.0,
        ge=0.0,
        description="Unix timestamp when the cache entry was created.",
    )


CacheEntry = CachedResponse


# ---------------------------------------------------------------------------
# Vector Packing & Math Helpers
# ---------------------------------------------------------------------------


def pack_float32_vector(vector: Sequence[float]) -> bytes:
    """Serialize a float vector into little-endian IEEE 754 FLOAT32 bytes."""
    if not vector:
        return b""
    return struct.pack(f"<{len(vector)}f", *(float(v) for v in vector))


def unpack_float32_vector(raw: bytes) -> list[float]:
    """Deserialize little-endian IEEE 754 FLOAT32 bytes into a float list."""
    if not raw:
        return []
    if len(raw) % 4 != 0:
        raise ValueError(f"FLOAT32 byte length must be divisible by 4, got {len(raw)}.")
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}f", raw))


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors, clamped to [0, 1]."""
    if len(vec_a) != len(vec_b) or not vec_a:
        raise ValueError("Vectors must be non-empty and have matching dimensions.")
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return round(max(0.0, min(1.0, sim)), 6)


# ---------------------------------------------------------------------------
# Private Parsing & Settings Helpers
# ---------------------------------------------------------------------------


def _escape_tag_value(value: str) -> str:
    """Escape special RediSearch TAG characters in namespace filter values."""
    return _TAG_SPECIAL_RE.sub(r"\\\1", value)


def _decode_str(raw: Any, default: str = "") -> str:
    if raw is None:
        return default
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _parse_metadata_json(raw: Any) -> dict[str, Any]:
    text = _decode_str(raw, "")
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_ft_search_top_hit(raw_result: Any) -> dict[str, str] | None:
    """Extract the top hit fields from RESP2 list or RESP3 dict ``FT.SEARCH`` output."""
    if not raw_result:
        return None

    # RESP3 dictionary format (optional)
    if isinstance(raw_result, dict):
        results = raw_result.get("results") or []
        if not results:
            return None
        first = results[0]
        attrs = first.get("extra_attributes") if isinstance(first, dict) else None
        if not isinstance(attrs, dict):
            return None
        return {_decode_str(k): _decode_str(v) for k, v in attrs.items()}

    # Standard RESP2 array format: [total_count, key_1, [field_1, val_1, ...], ...]
    if isinstance(raw_result, (list, tuple)):
        if len(raw_result) < 3 or int(raw_result[0]) <= 0:
            return None
        raw_fields = raw_result[2]
        if isinstance(raw_fields, dict):
            return {_decode_str(k): _decode_str(v) for k, v in raw_fields.items()}
        if isinstance(raw_fields, (list, tuple)):
            fields: dict[str, str] = {}
            for idx in range(0, len(raw_fields) - 1, 2):
                key = _decode_str(raw_fields[idx])
                val = _decode_str(raw_fields[idx + 1])
                fields[key] = val
            return fields

    return None


def _try_get_settings() -> Settings | None:
    """Resolve Settings safely without crashing in keyless offline unit tests."""
    try:
        return get_settings()
    except ValidationError:
        return None


def _is_missing_index_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "unknown index name" in message
        or "no such index" in message
        or "index not found" in message
    )


def _is_existing_index_error(exc: Exception) -> bool:
    return "index already exists" in str(exc).lower()


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class SemanticCache:
    """Async Redis 8 RediSearch vector semantic cache with fail-open degradation."""

    def __init__(
        self,
        redis_client: Any,
        embedder: DenseEmbedder,
        *,
        settings: Settings | None = None,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = None,
        index_name: str = DEFAULT_INDEX_NAME,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        default_namespace: str = DEFAULT_NAMESPACE,
        vector_dim: int = DENSE_DIM,
        algorithm: VectorIndexAlgorithm | str = VectorIndexAlgorithm.HNSW,
    ) -> None:
        resolved_settings = settings if settings is not None else _try_get_settings()

        if similarity_threshold is not None:
            resolved_threshold = float(similarity_threshold)
        elif resolved_settings is not None:
            resolved_threshold = float(resolved_settings.cache_similarity_threshold)
        else:
            resolved_threshold = DEFAULT_SIMILARITY_THRESHOLD

        if ttl_seconds is not None:
            resolved_ttl = int(ttl_seconds)
        elif resolved_settings is not None:
            resolved_ttl = int(resolved_settings.cache_ttl_seconds)
        else:
            resolved_ttl = DEFAULT_TTL_SECONDS

        if not 0.0 <= resolved_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be within [0.0, 1.0], got {resolved_threshold}."
            )
        if resolved_ttl <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {resolved_ttl}.")
        if vector_dim <= 0:
            raise ValueError(f"vector_dim must be positive, got {vector_dim}.")

        self._redis = redis_client
        self._embedder = embedder
        self._similarity_threshold = resolved_threshold
        self._ttl_seconds = resolved_ttl
        self._index_name = index_name
        self._key_prefix = key_prefix.rstrip(":")
        self._default_namespace = default_namespace
        self._vector_dim = vector_dim
        self._algorithm = (
            algorithm
            if isinstance(algorithm, VectorIndexAlgorithm)
            else VectorIndexAlgorithm(algorithm.strip().upper())
        )
        self._index_ready = False
        self.singleflight = SingleFlight()

    @classmethod
    def from_url(
        cls,
        embedder: DenseEmbedder,
        *,
        redis_url: str | None = None,
        settings: Settings | None = None,
        **kwargs: Any,
    ) -> SemanticCache:
        """Construct a ``SemanticCache`` connected via ``redis.asyncio.Redis.from_url``."""
        resolved_settings = settings if settings is not None else _try_get_settings()
        target_url = (
            redis_url
            or (resolved_settings.redis_url if resolved_settings is not None else None)
            or DEFAULT_REDIS_URL
        )
        client = Redis.from_url(target_url, decode_responses=False)
        return cls(client, embedder, settings=resolved_settings, **kwargs)

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    @property
    def index_name(self) -> str:
        return self._index_name

    def _build_entry_key(self, namespace: str, query: str) -> str:
        digest = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:24]
        return f"{self._key_prefix}:{namespace}:{digest}"

    async def _embed_query(self, query: str) -> list[float]:
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(None, self._embedder.embed, [query])
        if not vectors or len(vectors[0]) != self._vector_dim:
            got_dim = len(vectors[0]) if vectors else 0
            raise ValueError(
                f"Expected {self._vector_dim}-dim embedding from embedder, got {got_dim}."
            )
        return [float(v) for v in vectors[0]]

    async def ensure_index(self) -> bool:
        """Idempotently create the RediSearch HASH + VECTOR index in Redis 8."""
        if self._index_ready:
            return True

        try:
            await self._redis.execute_command("FT.INFO", self._index_name)
            self._index_ready = True
            return True
        except Exception as info_exc:
            if not _is_missing_index_error(info_exc):
                logger.warning(
                    "Redis semantic cache FT.INFO failed (fail-open)",
                    index_name=self._index_name,
                    error_type=type(info_exc).__name__,
                    error=str(info_exc),
                )
                return False

        try:
            await self._redis.execute_command(
                "FT.CREATE",
                self._index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                f"{self._key_prefix}:",
                "SCHEMA",
                "namespace",
                "TAG",
                "query",
                "TEXT",
                "response",
                "TEXT",
                "metadata",
                "TEXT",
                "created_at",
                "NUMERIC",
                "embedding",
                "VECTOR",
                self._algorithm.value,
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self._vector_dim),
                "DISTANCE_METRIC",
                "COSINE",
            )
            self._index_ready = True
            logger.info(
                "Created Redis vector semantic cache index",
                index_name=self._index_name,
                algorithm=self._algorithm.value,
                vector_dim=self._vector_dim,
            )
            return True
        except Exception as create_exc:
            if _is_existing_index_error(create_exc):
                self._index_ready = True
                return True
            logger.warning(
                "Redis semantic cache FT.CREATE failed (fail-open)",
                index_name=self._index_name,
                error_type=type(create_exc).__name__,
                error=str(create_exc),
            )
            return False

    async def get(
        self,
        query: str,
        *,
        namespace: str | None = None,
        threshold: float | None = None,
    ) -> CachedResponse | None:
        """Look up a semantically similar cached response for ``query``.

        Updates ``RequestMetrics.cache_hit`` in the active async ``contextvars``
        scope. Never raises on Redis or embedder failures (fail-open).
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            update_request_metrics(cache_hit=False)
            return None

        active_ns = namespace or self._default_namespace
        active_threshold = self._similarity_threshold if threshold is None else float(threshold)

        try:
            if not await self.ensure_index():
                update_request_metrics(cache_hit=False)
                return None

            query_vec = await self._embed_query(cleaned_query)
            packed_vec = pack_float32_vector(query_vec)
            escaped_ns = _escape_tag_value(active_ns)
            ft_query = f"(@namespace:{{{escaped_ns}}})=>[KNN 1 @embedding $vec AS vector_distance]"

            raw_result = await self._redis.execute_command(
                "FT.SEARCH",
                self._index_name,
                ft_query,
                "PARAMS",
                "2",
                "vec",
                packed_vec,
                "SORTBY",
                "vector_distance",
                "ASC",
                "RETURN",
                "6",
                "query",
                "response",
                "namespace",
                "metadata",
                "created_at",
                "vector_distance",
                "DIALECT",
                "2",
            )

            top_hit = _parse_ft_search_top_hit(raw_result)
            if not top_hit:
                update_request_metrics(cache_hit=False)
                return None

            raw_distance = top_hit.get("vector_distance")
            if raw_distance is None:
                update_request_metrics(cache_hit=False)
                return None

            similarity = round(max(0.0, min(1.0, 1.0 - float(raw_distance))), 6)
            if similarity < active_threshold:
                update_request_metrics(cache_hit=False)
                logger.debug(
                    "Semantic cache miss below threshold",
                    namespace=active_ns,
                    similarity=similarity,
                    threshold=active_threshold,
                )
                return None

            cached_response_text = top_hit.get("response", "")
            parsed_meta = _parse_metadata_json(top_hit.get("metadata"))
            if not is_cacheable_response(cached_response_text, metadata=parsed_meta):
                update_request_metrics(cache_hit=False)
                logger.warning(
                    "Ignored polluted or fallback entry during semantic cache lookup",
                    namespace=active_ns,
                )
                return None

            update_request_metrics(cache_hit=True)
            created_at_raw = top_hit.get("created_at", "0.0")
            try:
                created_at = max(0.0, float(created_at_raw))
            except ValueError:
                created_at = 0.0

            return CachedResponse(
                query=top_hit.get("query", cleaned_query),
                response=cached_response_text,
                namespace=top_hit.get("namespace", active_ns),
                similarity=similarity,
                metadata=parsed_meta,
                created_at=created_at,
            )
        except (RedisError, OSError, TimeoutError, ValueError) as exc:
            update_request_metrics(cache_hit=False)
            logger.warning(
                "Semantic cache lookup failed (fail-open)",
                namespace=active_ns,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None
        except Exception as exc:
            update_request_metrics(cache_hit=False)
            logger.warning(
                "Unexpected semantic cache lookup failure (fail-open)",
                namespace=active_ns,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None

    async def set(
        self,
        query: str,
        response: str,
        *,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> str | None:
        """Store a query embedding and response in Redis with TTL expiration.

        Returns the stored Redis key on success, or ``None`` if writing failed
        (fail-open).
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return None

        active_ns = namespace or self._default_namespace
        if not is_cacheable_response(response, metadata=metadata):
            logger.warning(
                "Blocked fallback or HTML-polluted response from semantic cache write",
                namespace=active_ns,
            )
            return None

        active_ttl = self._ttl_seconds if ttl_seconds is None else int(ttl_seconds)

        try:
            if not await self.ensure_index():
                return None

            query_vec = await self._embed_query(cleaned_query)
            packed_vec = pack_float32_vector(query_vec)
            redis_key = self._build_entry_key(active_ns, cleaned_query)
            mapping: dict[str, str | bytes] = {
                "namespace": active_ns,
                "query": cleaned_query,
                "response": response,
                "metadata": json.dumps(metadata or {}, ensure_ascii=False),
                "created_at": str(round(time.time(), 3)),
                "embedding": packed_vec,
            }
            await self._redis.hset(redis_key, mapping=mapping)
            if active_ttl > 0:
                await self._redis.expire(redis_key, active_ttl)
            return redis_key
        except (RedisError, OSError, TimeoutError, ValueError) as exc:
            logger.warning(
                "Semantic cache write failed (fail-open)",
                namespace=active_ns,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None
        except Exception as exc:
            logger.warning(
                "Unexpected semantic cache write failure (fail-open)",
                namespace=active_ns,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None

    async def invalidate_namespace(self, namespace: str | None = None) -> int:
        """Delete all cached entries belonging to ``namespace`` (fail-open)."""
        active_ns = namespace or self._default_namespace
        pattern = f"{self._key_prefix}:{active_ns}:*"
        try:
            keys: list[Any] = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)
            if not keys:
                return 0
            deleted = await self._redis.delete(*keys)
            return int(deleted)
        except Exception as exc:
            logger.warning(
                "Semantic cache namespace invalidation failed (fail-open)",
                namespace=active_ns,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return 0

    async def aclose(self) -> None:
        """Close the underlying async Redis connection gracefully."""
        close_fn = getattr(self._redis, "aclose", None)
        if callable(close_fn):
            await close_fn()
            return
        legacy_close = getattr(self._redis, "close", None)
        if callable(legacy_close):
            res = legacy_close()
            if asyncio.iscoroutine(res):
                await res

    # Convenience aliases aligned with common cache APIs
    lookup = get
    put = set
    store = set


RedisSemanticCache = SemanticCache


# ---------------------------------------------------------------------------
# SingleFlight & In-Memory Vector Semantic Cache (Demo / Redis Fallback)
# ---------------------------------------------------------------------------


class SingleFlight:
    """Deduplicates concurrent async work sharing the same logical key."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._inflight: dict[str, asyncio.Future[Any]] = {}

    async def do(self, key: str, fn: Callable[[], Awaitable[_T]]) -> _T:
        """Execute ``fn()`` once for ``key`` among concurrent callers."""
        async with self._guard:
            existing = self._inflight.get(key)
            if existing is not None:
                return await asyncio.shield(existing)  # type: ignore[no-any-return]
            loop = asyncio.get_running_loop()
            future: asyncio.Future[_T] = loop.create_future()
            self._inflight[key] = future
            is_leader = True

        if not is_leader:  # pragma: no cover - defensive clarity
            return await asyncio.shield(future)

        try:
            result = await fn()
            if not future.done():
                future.set_result(result)
            return result
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            async with self._guard:
                if self._inflight.get(key) is future:
                    self._inflight.pop(key, None)


class _HashingTextEmbedder:
    """Zero-GPU deterministic 256-dim L2-normalized hashed n-gram embedder."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            cleaned = re.sub(r"\s+", " ", text.strip().lower())
            if not cleaned:
                vectors.append(vec)
                continue

            words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]|[\uac00-\ud7af]+", cleaned)
            tokens: list[str] = list(words)
            for word in words:
                if len(word) >= 3:
                    tokens.extend(word[i : i + 3] for i in range(len(word) - 2))
            if not tokens:
                tokens = [cleaned]

            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self._dim
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                vec[idx] += sign

            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0.0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    def get_vram_usage_mb(self) -> float:
        return 0.0


class InMemorySemanticCache:
    """In-memory vector semantic cache with TTL expiry and SingleFlight protection.

    Used in ``demo`` / ``test`` modes and as an automatic fail-open fallback
    when Redis 8 is unreachable in ``live`` mode.
    """

    def __init__(
        self,
        *,
        embedder: DenseEmbedder | None = None,
        settings: Settings | None = None,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = None,
        default_namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        resolved_settings = settings if settings is not None else _try_get_settings()
        if similarity_threshold is not None:
            resolved_threshold = float(similarity_threshold)
        elif resolved_settings is not None:
            resolved_threshold = float(resolved_settings.cache_similarity_threshold)
        else:
            resolved_threshold = DEFAULT_SIMILARITY_THRESHOLD

        if ttl_seconds is not None:
            resolved_ttl = int(ttl_seconds)
        elif resolved_settings is not None:
            resolved_ttl = int(resolved_settings.cache_ttl_seconds)
        else:
            resolved_ttl = DEFAULT_TTL_SECONDS

        if not 0.0 <= resolved_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be within [0.0, 1.0], got {resolved_threshold}."
            )
        if resolved_ttl <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {resolved_ttl}.")

        self._embedder: DenseEmbedder = embedder or _HashingTextEmbedder()
        self._similarity_threshold = resolved_threshold
        self._ttl_seconds = resolved_ttl
        self._default_namespace = default_namespace
        self._entries: dict[tuple[str, str], tuple[list[float], CachedResponse, float]] = {}
        self._lock = asyncio.Lock()
        self.singleflight = SingleFlight()

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def _purge_expired_locked(self, now: float) -> None:
        expired = [k for k, (_, _, expires_at) in self._entries.items() if expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)

    async def get(
        self,
        query: str,
        *,
        namespace: str | None = None,
        threshold: float | None = None,
    ) -> CachedResponse | None:
        """Look up the best-matching cached response above cosine threshold."""
        cleaned_query = query.strip()
        if not cleaned_query:
            update_request_metrics(cache_hit=False)
            return None

        active_ns = namespace or self._default_namespace
        active_threshold = self._similarity_threshold if threshold is None else float(threshold)
        query_vec = self._embedder.embed([cleaned_query])[0]

        async def _lookup() -> CachedResponse | None:
            now = time.monotonic()
            async with self._lock:
                self._purge_expired_locked(now)
                best_sim = -1.0
                best_entry: CachedResponse | None = None
                for (entry_ns, _), (entry_vec, entry, _) in self._entries.items():
                    if entry_ns != active_ns:
                        continue
                    sim = cosine_similarity(query_vec, entry_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

            if best_entry is None or best_sim < active_threshold:
                return None
            return best_entry.model_copy(update={"similarity": best_sim})

        flight_key = f"get:{active_ns}:{cleaned_query.lower()}"
        hit = await self.singleflight.do(flight_key, _lookup)
        if hit is None or not is_cacheable_response(hit.response, metadata=hit.metadata):
            update_request_metrics(cache_hit=False)
            return None

        update_request_metrics(cache_hit=True)
        return hit

    async def set(
        self,
        query: str,
        response: str,
        *,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> str | None:
        """Store a query vector and response with TTL expiration."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return None

        active_ns = namespace or self._default_namespace
        if not is_cacheable_response(response, metadata=metadata):
            logger.warning(
                "Blocked fallback or HTML-polluted response from semantic cache write",
                namespace=active_ns,
            )
            return None

        active_ttl = self._ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        query_vec = self._embedder.embed([cleaned_query])[0]
        digest = hashlib.sha256(cleaned_query.encode("utf-8")).hexdigest()[:24]
        cache_key = f"{DEFAULT_KEY_PREFIX}:{active_ns}:{digest}"

        entry = CachedResponse(
            query=cleaned_query,
            response=response,
            namespace=active_ns,
            similarity=1.0,
            metadata=dict(metadata or {}),
            created_at=round(time.time(), 3),
        )
        expires_at = time.monotonic() + max(1, active_ttl)

        async with self._lock:
            self._purge_expired_locked(time.monotonic())
            self._entries[(active_ns, cleaned_query)] = (query_vec, entry, expires_at)

        return cache_key

    async def invalidate_namespace(self, namespace: str | None = None) -> int:
        """Remove all cached entries belonging to ``namespace``."""
        active_ns = namespace or self._default_namespace
        async with self._lock:
            keys = [k for k in self._entries if k[0] == active_ns]
            for k in keys:
                self._entries.pop(k, None)
            return len(keys)

    async def aclose(self) -> None:
        """Clear in-memory entries on shutdown."""
        async with self._lock:
            self._entries.clear()

    lookup = get
    put = set
    store = set


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__: Sequence[str] = [
    "DEFAULT_INDEX_NAME",
    "DEFAULT_KEY_PREFIX",
    "DEFAULT_NAMESPACE",
    "DEFAULT_REDIS_URL",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_TTL_SECONDS",
    "CacheEntry",
    "CachedResponse",
    "InMemorySemanticCache",
    "RedisSemanticCache",
    "SemanticCache",
    "SingleFlight",
    "VectorIndexAlgorithm",
    "contains_raw_html_noise",
    "cosine_similarity",
    "is_cacheable_response",
    "pack_float32_vector",
    "unpack_float32_vector",
]
