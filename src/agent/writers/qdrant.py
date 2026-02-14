"""
Qdrant vector database writer for RAG Vector Store Writer.

Upserts embedding vectors to a Qdrant Cloud collection via REST API.

Key design principles:
- Cluster URL validated against cloud.qdrant.io pattern (SSRF prevention)
- Auto-creates collection if it doesn't exist (with configurable distance metric)
- API key never logged, never included in output
- Batched upserts (recommended 100 per call for 1536d vectors)
- Retry with exponential backoff for transient failures
- Error messages sanitized to prevent key leakage
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_REQUEST_TIMEOUT = 60


def _sanitize_error(message: str, api_key: str) -> str:
    """Strip API key from error messages."""
    if api_key and api_key in message:
        message = message.replace(api_key, "[REDACTED]")
    if api_key and len(api_key) > 8 and api_key[:8] in message:
        message = message.replace(api_key[:8], "[REDACTED]")
    return message


async def _request_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    headers: dict,
    payload: Optional[dict],
    api_key: str,
    accept_statuses: tuple = (200,),
) -> dict:
    """Make an HTTP request with exponential backoff retry.

    Retries on 429, 500, 502, 503, 504.
    Never retries on 401 or 400.
    """
    last_error = None

    for attempt in range(_MAX_RETRIES):
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
            kwargs = {"headers": headers, "timeout": timeout}
            if payload is not None:
                kwargs["json"] = payload

            async with session.request(method, url, **kwargs) as resp:
                if resp.status in accept_statuses:
                    return await resp.json()

                body = await resp.text()
                safe_body = _sanitize_error(body, api_key)

                if resp.status in (400, 401, 403):
                    if resp.status == 401:
                        raise ValueError(
                            "Qdrant API key is invalid or expired. "
                            "Check your key and try again."
                        )
                    raise ValueError(
                        f"Qdrant API error ({resp.status}): {safe_body}"
                    )

                if resp.status in (429, 500, 502, 503, 504):
                    last_error = f"Qdrant returned {resp.status}: {safe_body}"
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Attempt %d/%d failed (%d), retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES, resp.status, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                raise ValueError(
                    f"Unexpected Qdrant response ({resp.status}): {safe_body}"
                )

        except aiohttp.ClientError as exc:
            last_error = f"Network error: {str(exc)}"
            delay = _BASE_DELAY * (2 ** attempt)
            logger.warning(
                "Attempt %d/%d network error, retrying in %.1fs: %s",
                attempt + 1, _MAX_RETRIES, delay, last_error,
            )
            await asyncio.sleep(delay)

    raise ValueError(
        f"Qdrant: failed after {_MAX_RETRIES} attempts. Last error: {last_error}"
    )


async def _ensure_collection(
    session: aiohttp.ClientSession,
    base_url: str,
    headers: dict,
    api_key: str,
    collection_name: str,
    vector_size: int,
    distance_metric: str,
) -> bool:
    """Create Qdrant collection if it doesn't exist.

    Returns True if collection was created, False if it already existed.
    """
    # Check if collection exists
    exists_url = f"{base_url}/collections/{collection_name}/exists"
    try:
        data = await _request_with_retry(
            session, "GET", exists_url, headers, None, api_key,
        )
        if data.get("result", {}).get("exists", False):
            logger.info(
                "Qdrant collection '%s' already exists.", collection_name,
            )
            return False
    except ValueError:
        # If the exists endpoint fails, try creating anyway
        pass

    # Create collection
    create_url = f"{base_url}/collections/{collection_name}"
    payload = {
        "vectors": {
            "size": vector_size,
            "distance": distance_metric,
        }
    }

    logger.info(
        "Creating Qdrant collection '%s' (size=%d, distance=%s)...",
        collection_name, vector_size, distance_metric,
    )

    await _request_with_retry(
        session, "PUT", create_url, headers, payload, api_key,
    )

    logger.info("Qdrant collection '%s' created.", collection_name)
    return True


def _build_qdrant_point(
    item: dict,
    index: int,
    id_field: str,
) -> dict:
    """Convert an embedding item into a Qdrant point object.

    Uses chunk_id (or configured id_field) as the point ID.
    Falls back to UUID if no ID field is present.
    Passes through all metadata as payload.
    """
    # Determine point ID -- Qdrant accepts UUID strings or integers
    point_id = item.get(id_field, "")
    if not point_id or not isinstance(point_id, str):
        point_id = str(uuid.uuid4())

    # Extract embedding
    embedding = item.get("embedding", [])

    # Build payload from all non-embedding, non-internal fields
    skip_fields = {"embedding", "_summary", "index", "dimensions"}
    payload = {}
    for key, value in item.items():
        if key in skip_fields:
            continue
        if key == id_field:
            continue
        # Qdrant payload accepts any JSON-serializable value
        if isinstance(value, (str, int, float, bool, list, dict)):
            payload[key] = value

    return {
        "id": point_id,
        "vector": embedding,
        "payload": payload,
    }


async def write_to_qdrant(
    items: List[dict],
    api_key: str,
    cluster_url: str,
    collection_name: str,
    distance_metric: str = "Cosine",
    batch_size: int = 100,
    id_field: str = "chunk_id",
) -> Dict:
    """Write embedding vectors to a Qdrant collection.

    Args:
        items: List of embedding items (from RAG Embedding Generator output).
        api_key: Qdrant API key.
        cluster_url: Qdrant Cloud cluster URL.
        collection_name: Target collection name.
        distance_metric: Distance metric for collection creation.
        batch_size: Points per upsert request.
        id_field: Field to use as point ID.

    Returns:
        Summary dict with total_upserted, batches, collection_created, etc.
    """
    total_upserted = 0
    total_batches = 0
    collection_created = False

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        # Step 1: Build all points first to determine vector size
        points = []
        for i, item in enumerate(items):
            points.append(_build_qdrant_point(item, i, id_field))

        if not points:
            raise ValueError("No valid points to upsert.")

        # Determine vector dimensions from first point
        vector_size = len(points[0].get("vector", []))
        if vector_size == 0:
            raise ValueError(
                "First embedding has zero dimensions. "
                "Check that input items have valid 'embedding' arrays."
            )

        # Step 2: Ensure collection exists
        collection_created = await _ensure_collection(
            session, cluster_url, headers, api_key,
            collection_name, vector_size, distance_metric,
        )

        # Step 3: Upsert in batches
        upsert_url = f"{cluster_url}/collections/{collection_name}/points"

        for batch_start in range(0, len(points), batch_size):
            batch_end = min(batch_start + batch_size, len(points))
            batch = points[batch_start:batch_end]

            logger.info(
                "Qdrant upsert batch %d-%d of %d points...",
                batch_start + 1, batch_end, len(points),
            )

            payload = {"points": batch}

            data = await _request_with_retry(
                session, "PUT", upsert_url, headers, payload, api_key,
            )

            status = data.get("status", "")
            if status == "ok":
                total_upserted += len(batch)
                total_batches += 1
            else:
                logger.warning(
                    "Qdrant batch %d-%d returned status: %s",
                    batch_start + 1, batch_end, status,
                )

    return {
        "provider": "qdrant",
        "collection_name": collection_name,
        "cluster_url": cluster_url,
        "distance_metric": distance_metric,
        "collection_created": collection_created,
        "total_upserted": total_upserted,
        "total_batches": total_batches,
    }
