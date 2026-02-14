"""
Pinecone vector database writer for RAG Vector Store Writer.

Upserts embedding vectors to a Pinecone index via REST API.

Key design principles:
- Control plane URL hardcoded to api.pinecone.io (SSRF prevention)
- Data plane URL resolved from control plane, then cached
- API key never logged, never included in output
- Batched upserts (max 1000 per call, recommended 100-200 for 1536d)
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

# Hardcoded control plane URL -- the ONLY URL used to resolve index hosts
_CONTROL_PLANE_URL = "https://api.pinecone.io"

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
) -> dict:
    """Make an HTTP request with exponential backoff retry.

    Retries on 429, 500, 502, 503, 504.
    Never retries on 401 (auth) or 400 (bad request).
    """
    last_error = None

    for attempt in range(_MAX_RETRIES):
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
            kwargs = {"headers": headers, "timeout": timeout}
            if payload is not None:
                kwargs["json"] = payload

            async with session.request(method, url, **kwargs) as resp:
                if resp.status == 200:
                    return await resp.json()

                body = await resp.text()
                safe_body = _sanitize_error(body, api_key)

                if resp.status in (400, 401, 403):
                    if resp.status == 401:
                        raise ValueError(
                            "Pinecone API key is invalid or expired. "
                            "Check your key and try again."
                        )
                    raise ValueError(
                        f"Pinecone API error ({resp.status}): {safe_body}"
                    )

                if resp.status in (429, 500, 502, 503, 504):
                    last_error = f"Pinecone returned {resp.status}: {safe_body}"
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Attempt %d/%d failed (%d), retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES, resp.status, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                raise ValueError(
                    f"Unexpected Pinecone response ({resp.status}): {safe_body}"
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
        f"Pinecone: failed after {_MAX_RETRIES} attempts. Last error: {last_error}"
    )


async def _resolve_index_host(
    session: aiohttp.ClientSession,
    api_key: str,
    index_name: str,
) -> str:
    """Resolve the data plane host for a Pinecone index.

    Calls the control plane API (hardcoded URL) to get the index host.
    """
    url = f"{_CONTROL_PLANE_URL}/indexes/{index_name}"
    headers = {
        "Api-Key": api_key,
        "X-Pinecone-Api-Version": "2024-07",
    }

    data = await _request_with_retry(
        session, "GET", url, headers, None, api_key
    )

    host = data.get("host")
    if not host:
        raise ValueError(
            f"Could not resolve host for Pinecone index '{index_name}'. "
            f"Verify the index exists in your Pinecone project."
        )

    logger.info("Resolved Pinecone index '%s' -> host '%s'", index_name, host)
    return host


def _build_pinecone_vector(
    item: dict,
    index: int,
    id_field: str,
) -> dict:
    """Convert an embedding item into a Pinecone vector object.

    Uses chunk_id (or configured id_field) as the vector ID.
    Falls back to UUID if no ID field is present.
    Passes through all metadata fields except embedding and _summary.
    """
    # Determine vector ID
    vec_id = item.get(id_field, "")
    if not vec_id or not isinstance(vec_id, str):
        vec_id = str(uuid.uuid4())

    # Extract embedding
    embedding = item.get("embedding", [])

    # Build metadata from all non-embedding, non-internal fields
    skip_fields = {"embedding", "_summary", "index", "dimensions"}
    metadata = {}
    for key, value in item.items():
        if key in skip_fields:
            continue
        if key == id_field:
            continue
        # Pinecone metadata values: str, int, float, bool, list of str
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            metadata[key] = value

    return {
        "id": vec_id,
        "values": embedding,
        "metadata": metadata,
    }


async def write_to_pinecone(
    items: List[dict],
    api_key: str,
    index_name: str,
    namespace: str = "",
    batch_size: int = 100,
    id_field: str = "chunk_id",
) -> Dict:
    """Write embedding vectors to a Pinecone index.

    Args:
        items: List of embedding items (from RAG Embedding Generator output).
        api_key: Pinecone API key.
        index_name: Pinecone index name.
        namespace: Target namespace (empty = default).
        batch_size: Vectors per upsert request (max 1000).
        id_field: Field to use as vector ID.

    Returns:
        Summary dict with total_upserted, batches, etc.
    """
    total_upserted = 0
    total_batches = 0

    async with aiohttp.ClientSession() as session:
        # Step 1: Resolve the index host
        host = await _resolve_index_host(session, api_key, index_name)
        base_url = f"https://{host}"

        headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "X-Pinecone-Api-Version": "2024-07",
        }

        # Step 2: Build all vectors
        vectors = []
        for i, item in enumerate(items):
            vectors.append(_build_pinecone_vector(item, i, id_field))

        # Step 3: Upsert in batches
        for batch_start in range(0, len(vectors), batch_size):
            batch_end = min(batch_start + batch_size, len(vectors))
            batch = vectors[batch_start:batch_end]

            logger.info(
                "Pinecone upsert batch %d-%d of %d vectors...",
                batch_start + 1, batch_end, len(vectors),
            )

            payload = {"vectors": batch}
            if namespace:
                payload["namespace"] = namespace

            data = await _request_with_retry(
                session, "POST",
                f"{base_url}/vectors/upsert",
                headers, payload, api_key,
            )

            count = data.get("upsertedCount", 0)
            total_upserted += count
            total_batches += 1

    return {
        "provider": "pinecone",
        "index_name": index_name,
        "namespace": namespace or "(default)",
        "total_upserted": total_upserted,
        "total_batches": total_batches,
    }
