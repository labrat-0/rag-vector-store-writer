"""
Main entrypoint for RAG Vector Store Writer Apify Actor.

Supports two input modes:
1. Dataset chaining: User provides a dataset_id from RAG Embedding Generator.
   Loads items, filters out _summary rows, routes to the selected provider.
2. Raw vectors: User provides a vectors array directly in the input JSON.

Output: A dataset item per batch with upsert results, plus a summary item
with totals, billing, and provider details.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

from apify import Actor

from .pricing import calculate_billing
from .validation import validate_input, MAX_DATASET_ITEMS
from .writers.pinecone import write_to_pinecone
from .writers.qdrant import write_to_qdrant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _load_dataset(dataset_id: str) -> List[dict]:
    """Load embedding items from an Apify dataset.

    Filters out _summary rows from RAG Embedding Generator output.
    Validates that items have an 'embedding' field.

    Returns list of embedding item dicts.
    """
    try:
        dataset = await Actor.open_dataset(id=dataset_id)
        list_result = await dataset.get_data(limit=MAX_DATASET_ITEMS)
        items = list_result.items if list_result else []
    except Exception as exc:
        logger.error("Failed to open dataset '%s': %s", dataset_id, exc)
        raise ValueError(
            f"Could not load dataset '{dataset_id}'. "
            f"Verify the dataset ID exists and this actor has access to it."
        ) from exc

    if not items:
        raise ValueError(
            f"Dataset '{dataset_id}' is empty or contains no items."
        )

    # Filter out summary items from RAG Embedding Generator
    embedding_items = [
        item for item in items
        if isinstance(item, dict) and not item.get("_summary", False)
    ]

    if not embedding_items:
        raise ValueError(
            f"Dataset '{dataset_id}' contains no embedding items "
            f"(only summary rows found)."
        )

    # Validate that items have embeddings
    valid_items = []
    skipped = 0
    for item in embedding_items:
        emb = item.get("embedding")
        if isinstance(emb, list) and len(emb) > 0:
            valid_items.append(item)
        else:
            skipped += 1

    if not valid_items:
        raise ValueError(
            f"No items with valid 'embedding' arrays found in dataset "
            f"'{dataset_id}'. Ensure the dataset was produced by "
            f"RAG Embedding Generator or contains items with "
            f"'embedding' fields."
        )

    if skipped > 0:
        logger.warning(
            "Skipped %d items without valid embeddings out of %d total.",
            skipped, len(embedding_items),
        )

    logger.info(
        "Loaded %d embedding items from dataset '%s'.",
        len(valid_items), dataset_id,
    )

    return valid_items


async def main() -> None:
    async with Actor:
        actor_input: Dict[str, Any] = await Actor.get_input() or {}

        # --- Validate all inputs ---
        validated, error = validate_input(actor_input)
        if error:
            await Actor.fail(status_message=error)
            return

        start_time = time.time()

        try:
            items: List[dict] = []

            # --- Dataset mode takes priority ---
            if validated.dataset_id:
                logger.info(
                    "Mode: dataset chaining (dataset_id=%s)",
                    validated.dataset_id,
                )
                items = await _load_dataset(validated.dataset_id)

            # --- Raw vectors mode ---
            elif validated.vectors:
                logger.info(
                    "Mode: raw vectors (%d items)", len(validated.vectors),
                )
                items = validated.vectors

            if not items:
                await Actor.fail(
                    status_message="No vectors to write after processing input."
                )
                return

            logger.info(
                "Writing %d vectors: provider=%s, index=%s, batch_size=%d",
                len(items), validated.provider,
                validated.index_name, validated.batch_size,
            )

            # --- Route to provider writer ---
            if validated.provider == "pinecone":
                result = await write_to_pinecone(
                    items=items,
                    api_key=validated.api_key,
                    index_name=validated.index_name,
                    namespace=validated.namespace,
                    batch_size=validated.batch_size,
                    id_field=validated.id_field,
                )
            elif validated.provider == "qdrant":
                result = await write_to_qdrant(
                    items=items,
                    api_key=validated.api_key,
                    cluster_url=validated.environment,
                    collection_name=validated.index_name,
                    distance_metric=validated.distance_metric,
                    batch_size=validated.batch_size,
                    id_field=validated.id_field,
                )
            else:
                await Actor.fail(
                    status_message=f"Unknown provider: {validated.provider}"
                )
                return

        except ValueError as exc:
            await Actor.fail(status_message=str(exc))
            return
        except Exception as exc:
            logger.exception("Unexpected error during vector write: %s", exc)
            await Actor.fail(
                status_message=(
                    f"Internal error: {type(exc).__name__}. Check logs."
                )
            )
            return

        duration = round(time.time() - start_time, 3)

        # --- Calculate billing ---
        total_vectors = result.get("total_upserted", 0)

        if total_vectors == 0:
            await Actor.fail(
                status_message=(
                    "No vectors were upserted. The provider accepted the "
                    "request but reported 0 vectors written. Check your "
                    "index/collection configuration."
                )
            )
            return

        billing = calculate_billing(total_vectors)

        # --- Push summary to dataset ---
        summary = {
            "_summary": True,
            "provider": validated.provider,
            "total_vectors_upserted": total_vectors,
            "total_batches": result.get("total_batches", 0),
            "processing_time": duration,
            "billing": billing,
        }

        # Add provider-specific details
        if validated.provider == "pinecone":
            summary["index_name"] = result.get("index_name", "")
            summary["namespace"] = result.get("namespace", "(default)")
        elif validated.provider == "qdrant":
            summary["collection_name"] = result.get("collection_name", "")
            summary["cluster_url"] = result.get("cluster_url", "")
            summary["distance_metric"] = result.get("distance_metric", "")
            summary["collection_created"] = result.get(
                "collection_created", False
            )

        await Actor.push_data(summary)

        logger.info(
            "Done: %d vectors upserted to %s in %d batches, "
            "%.3fs, $%.4f",
            total_vectors,
            validated.provider,
            result.get("total_batches", 0),
            duration,
            billing["amount"],
        )


if __name__ == "__main__":
    asyncio.run(main())
