# RAG Vector Store Writer Skill

## Description
This agent writes embedding vectors to Pinecone or Qdrant vector databases. It accepts dataset output from RAG Embedding Generator (via dataset_id) or raw vector arrays (via vectors input). Each item's embedding is upserted to the target index/collection with all metadata preserved as vector metadata (Pinecone) or payload (Qdrant). Handles batching, retries, collection auto-creation (Qdrant), and ID generation. Designed as the final storage step in a RAG pipeline.

## Inputs
- `api_key`: String (required). Pinecone or Qdrant API key. Marked `isSecret`, never logged or stored.
- `provider`: String (optional). `"pinecone"` (default) or `"qdrant"`.
- `index_name`: String (required). Pinecone index name or Qdrant collection name. Qdrant collections are auto-created if they don't exist.
- `environment`: String (optional). Qdrant Cloud cluster URL. Required for Qdrant. Must match `cloud.qdrant.io` pattern.
- `namespace`: String (optional). Pinecone namespace. Empty string for default. Ignored for Qdrant.
- `distance_metric`: String (optional). Qdrant distance metric for collection creation: `"Cosine"` (default), `"Dot"`, `"Euclid"`. Ignored for Pinecone.
- `dataset_id`: String (optional). Apify dataset ID from RAG Embedding Generator. Items must have `embedding` field. Takes priority over `vectors`.
- `vectors`: Array (optional). Direct vector input. Each item needs `embedding` (array of floats) and optionally `chunk_id` and metadata fields.
- `batch_size`: Integer (optional). Vectors per upsert request. Default: 100. Max: 1000 (Pinecone) or 500 (Qdrant).
- `id_field`: String (optional). Field to use as vector ID. Default: `"chunk_id"`. Falls back to UUID.

At least one of `dataset_id` or `vectors` must be provided, plus `api_key` and `index_name`.

## Outputs
A single summary item is pushed to the default dataset:
- `_summary`: Boolean. Always `true`.
- `provider`: String. Provider used (`pinecone` or `qdrant`).
- `total_vectors_upserted`: Integer. Total vectors written to the database.
- `total_batches`: Integer. Number of batch upsert requests made.
- `processing_time`: Float. Wall-clock seconds.
- `billing`: Object. Contains `total_vectors`, `amount`, and `rate_per_vector`.
- `index_name` / `collection_name`: String. Target index or collection.
- `namespace`: String. Pinecone namespace used (Pinecone only).
- `cluster_url`: String. Qdrant cluster URL (Qdrant only).
- `distance_metric`: String. Distance metric used (Qdrant only).
- `collection_created`: Boolean. Whether the collection was auto-created (Qdrant only).

## Pricing Model
- Pay-Per-Event: $0.0004 per vector ($0.40 per 1,000 vectors).
- This is the actor's platform fee. Users also pay the vector DB provider directly via their own API key.
- A typical 10-page website produces 50-100 vectors, costing $0.02-$0.04 actor fee.
- Billing is deterministic and included in the output summary.

## Constraints
- Maximum vectors per run: 50,000.
- Maximum dataset items: 50,000.
- Batch size: 1-1000 (Pinecone), 1-500 (Qdrant).
- Pinecone metadata limit: 40KB per record.
- Qdrant cluster URL must match `cloud.qdrant.io` pattern (SSRF prevention).
- Pinecone host resolved via hardcoded control plane only (`api.pinecone.io`).
- Index/collection names: alphanumeric with hyphens/underscores, 1-64 characters.
- API key never appears in output, logs, or error messages.
- Retry with exponential backoff on rate limits (429) and server errors (500-504), up to 3 attempts.
- Input sanitized: control characters stripped, IDs regex-validated, providers whitelisted.

## Example

Input:
```json
{
  "api_key": "pcsk_your-key",
  "provider": "pinecone",
  "index_name": "my-rag-index",
  "dataset_id": "abc123XYZ",
  "batch_size": 200
}
```

Output:
```json
{
  "_summary": true,
  "provider": "pinecone",
  "total_vectors_upserted": 42,
  "total_batches": 1,
  "processing_time": 2.341,
  "index_name": "my-rag-index",
  "namespace": "docs",
  "billing": {
    "total_vectors": 42,
    "amount": 0.0168,
    "rate_per_vector": 0.0004
  }
}
```

## Pipeline Position
```
Crawl (Website Content Crawler)
  -> Clean (optional)
    -> Chunk (RAG Content Chunker)
      -> Embed (RAG Embedding Generator)
        -> Store (this actor)
```
