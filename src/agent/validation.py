"""
Input validation and security layer for RAG Vector Store Writer.

Handles:
- API key presence validation (format varies by provider, so we check presence only)
- Provider whitelist enforcement
- Index/collection name validation (prevent injection)
- Qdrant cluster URL validation (prevent SSRF -- must match cloud.qdrant.io pattern)
- Dataset ID and field name validation (prevent injection)
- Batch size limits
- Vector format validation

Security model:
- API keys are validated for presence only, never logged or stored
- Qdrant cluster URLs validated against strict pattern (SSRF prevention)
- Pinecone host resolved via hardcoded control plane URL only
- All string inputs sanitized against control characters
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- Hard limits ---
MAX_VECTORS_COUNT = 50_000          # max vectors in a single run
MAX_DATASET_ITEMS = 50_000          # max items to process from a dataset
MAX_BATCH_SIZE_PINECONE = 1000      # Pinecone API limit
MAX_BATCH_SIZE_QDRANT = 500         # practical limit for Qdrant
MAX_METADATA_SIZE = 40_000          # 40KB Pinecone metadata limit per record

# --- Provider whitelist ---
VALID_PROVIDERS = {"pinecone", "qdrant"}

# --- Distance metrics for Qdrant ---
VALID_DISTANCE_METRICS = {"Cosine", "Dot", "Euclid"}

# --- Validation patterns ---
# Index/collection names: alphanumeric, hyphens, underscores, 1-64 chars
_INDEX_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# Qdrant Cloud URL pattern: https://{id}.{region}.{provider}.cloud.qdrant.io:6333
# Also allow custom port or no port for self-hosted
_QDRANT_URL_PATTERN = re.compile(
    r"^https://[a-zA-Z0-9._-]+\.cloud\.qdrant\.io(:\d+)?$"
)

# Dataset ID: same pattern as other actors
_DATASET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_~-]{0,63}$")

# ID field name: simple identifier
_FIELD_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]{0,63}$")

# Pinecone namespace: alphanumeric, hyphens, underscores, dots, or empty
_NAMESPACE_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{0,64}$")

# Control characters to strip
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class ValidatedInput:
    """Validated and sanitized input ready for processing."""

    api_key: str
    provider: str
    index_name: str
    environment: Optional[str]     # Qdrant cluster URL
    namespace: str                 # Pinecone namespace
    distance_metric: str           # Qdrant distance metric
    dataset_id: Optional[str]
    vectors: Optional[List[dict]]
    batch_size: int
    id_field: str


def sanitize_text(text: str) -> str:
    """Remove dangerous control characters."""
    return _CONTROL_CHARS.sub("", text)


def _sanitize_error(message: str, api_key: str) -> str:
    """Strip API key from error messages to prevent leakage."""
    if api_key and api_key in message:
        message = message.replace(api_key, "[REDACTED]")
    if api_key and len(api_key) > 8 and api_key[:8] in message:
        message = message.replace(api_key[:8], "[REDACTED]")
    return message


def validate_input(actor_input: dict) -> Tuple[Optional[ValidatedInput], Optional[str]]:
    """Validate and sanitize all input parameters.

    Returns:
        (ValidatedInput, None) on success.
        (None, error_message) on failure.
    """
    # --- Provider ---
    provider = actor_input.get("provider", "pinecone")
    if not isinstance(provider, str):
        provider = "pinecone"
    provider = provider.strip().lower()
    if provider not in VALID_PROVIDERS:
        return None, (
            f"Invalid provider '{provider}'. "
            f"Must be one of: {', '.join(sorted(VALID_PROVIDERS))}."
        )

    # --- API key ---
    api_key = actor_input.get("api_key", "")
    if not isinstance(api_key, str):
        api_key = ""
    api_key = api_key.strip()
    if not api_key:
        return None, (
            "API key is required. Provide your Pinecone or Qdrant API key."
        )

    # --- Index / Collection name ---
    index_name = actor_input.get("index_name", "")
    if not isinstance(index_name, str):
        index_name = ""
    index_name = index_name.strip()
    if not index_name:
        return None, (
            "Index/collection name is required. "
            "Pinecone: your index name. Qdrant: your collection name."
        )
    if not _INDEX_NAME_PATTERN.match(index_name):
        return None, (
            f"Invalid index/collection name: '{index_name}'. "
            f"Must be alphanumeric (with hyphens/underscores), 1-64 characters, "
            f"starting with a letter or digit."
        )

    # --- Qdrant cluster URL (required for Qdrant) ---
    environment = actor_input.get("environment")
    if provider == "qdrant":
        if not environment or not isinstance(environment, str):
            return None, (
                "Qdrant cluster URL is required. "
                "Provide your Qdrant Cloud URL "
                "(e.g., 'https://xyz.us-east-1.aws.cloud.qdrant.io:6333')."
            )
        environment = environment.strip().rstrip("/")
        if not _QDRANT_URL_PATTERN.match(environment):
            return None, (
                f"Invalid Qdrant cluster URL: '{environment}'. "
                f"Must match pattern: https://{{cluster}}.cloud.qdrant.io:6333"
            )
    else:
        environment = None

    # --- Namespace (Pinecone only) ---
    namespace = actor_input.get("namespace", "")
    if not isinstance(namespace, str):
        namespace = ""
    namespace = namespace.strip()
    if namespace and not _NAMESPACE_PATTERN.match(namespace):
        return None, (
            f"Invalid namespace: '{namespace}'. "
            f"Must be alphanumeric with hyphens/underscores/dots, max 64 characters."
        )

    # --- Distance metric (Qdrant only) ---
    distance_metric = actor_input.get("distance_metric", "Cosine")
    if not isinstance(distance_metric, str):
        distance_metric = "Cosine"
    if distance_metric not in VALID_DISTANCE_METRICS:
        return None, (
            f"Invalid distance metric '{distance_metric}'. "
            f"Must be one of: {', '.join(sorted(VALID_DISTANCE_METRICS))}."
        )

    # --- Input sources: dataset_id or vectors ---
    dataset_id = actor_input.get("dataset_id")
    vectors = actor_input.get("vectors")

    has_dataset = (
        dataset_id is not None
        and isinstance(dataset_id, str)
        and dataset_id.strip()
    )
    has_vectors = (
        vectors is not None
        and isinstance(vectors, list)
        and len(vectors) > 0
    )

    if not has_dataset and not has_vectors:
        return None, (
            "No input provided. Supply either 'dataset_id' (from RAG Embedding "
            "Generator output) or 'vectors' (raw vector array)."
        )

    # --- Validate dataset_id ---
    if has_dataset:
        dataset_id = dataset_id.strip()
        if not _DATASET_ID_PATTERN.match(dataset_id):
            return None, (
                f"Invalid dataset_id format: '{dataset_id}'. "
                f"Must be alphanumeric (with hyphens/underscores), 1-64 characters."
            )
    else:
        dataset_id = None

    # --- Validate raw vectors ---
    if has_vectors and not has_dataset:
        if len(vectors) > MAX_VECTORS_COUNT:
            return None, (
                f"Too many vectors: {len(vectors):,} provided, "
                f"maximum is {MAX_VECTORS_COUNT:,}."
            )
        for i, v in enumerate(vectors):
            if not isinstance(v, dict):
                return None, f"vectors[{i}] is not an object."
            if "embedding" not in v:
                return None, (
                    f"vectors[{i}] is missing 'embedding' field. "
                    f"Each vector must have an 'embedding' array of floats."
                )
            emb = v["embedding"]
            if not isinstance(emb, list) or not emb:
                return None, (
                    f"vectors[{i}]['embedding'] must be a non-empty array of numbers."
                )
    else:
        vectors = None

    # --- Validate batch_size ---
    batch_size = actor_input.get("batch_size", 100)
    if not isinstance(batch_size, int):
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            return None, f"batch_size must be an integer, got '{batch_size}'."

    max_batch = MAX_BATCH_SIZE_PINECONE if provider == "pinecone" else MAX_BATCH_SIZE_QDRANT
    if batch_size < 1 or batch_size > max_batch:
        return None, (
            f"batch_size must be between 1 and {max_batch} for {provider}, "
            f"got {batch_size}."
        )

    # --- Validate id_field ---
    id_field = actor_input.get("id_field", "chunk_id")
    if not isinstance(id_field, str):
        id_field = "chunk_id"
    id_field = id_field.strip()
    if not id_field:
        id_field = "chunk_id"
    if not _FIELD_NAME_PATTERN.match(id_field):
        return None, (
            f"Invalid id_field: '{id_field}'. "
            f"Must start with a letter or underscore, contain only "
            f"alphanumeric characters, underscores, or dots."
        )

    return ValidatedInput(
        api_key=api_key,
        provider=provider,
        index_name=index_name,
        environment=environment,
        namespace=namespace,
        distance_metric=distance_metric,
        dataset_id=dataset_id,
        vectors=vectors,
        batch_size=batch_size,
        id_field=id_field,
    ), None
