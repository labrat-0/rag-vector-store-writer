# RAG Vector Store Writer

Apify Actor that writes embedding vectors to Pinecone or Qdrant vector databases. Chains directly with RAG Embedding Generator output or accepts raw vectors with metadata. Handles batching, retries, collection creation, metadata mapping, and ID generation. Bring your own vector DB API key. MCP-ready for AI agent integration.

## Features
- Two vector database providers: Pinecone and Qdrant Cloud
- Two input modes: dataset chaining (from RAG Embedding Generator) or raw vector JSON
- Pinecone: resolves index host via control plane, batched upserts (max 1000/batch), namespace support
- Qdrant: auto-creates collection if missing (with configurable distance metric), batched upserts
- Metadata pass-through: all non-embedding fields from input become metadata/payload in the vector DB
- Configurable vector ID field (default: `chunk_id` from RAG Content Chunker), UUID fallback
- Exponential backoff retry on rate limits and transient failures (3 attempts)
- API key marked `isSecret` -- never logged, never stored, never included in output
- Hardcoded API URLs and URL pattern validation to prevent SSRF attacks
- Input validation and sanitization (provider whitelist, index name regex, batch size limits)

## Requirements
- Python 3.11+
- Apify platform account (for running as Actor)
- Pinecone or Qdrant Cloud API key and an existing index/cluster

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Actor Inputs
Defined in `.actor/INPUT_SCHEMA.json`:
- `api_key` (string, required) -- your Pinecone or Qdrant API key. Marked `isSecret`
- `provider` (string, optional) -- `"pinecone"` (default) or `"qdrant"`
- `index_name` (string, required) -- Pinecone index name or Qdrant collection name. Qdrant collections are auto-created if they don't exist
- `environment` (string, optional) -- Qdrant Cloud cluster URL (e.g., `https://xyz.us-east-1.aws.cloud.qdrant.io:6333`). Required for Qdrant
- `namespace` (string, optional) -- Pinecone namespace. Leave empty for default namespace. Ignored for Qdrant
- `distance_metric` (string, optional) -- Distance metric for Qdrant collection creation: `"Cosine"` (default), `"Dot"`, or `"Euclid"`. Only used when auto-creating a new collection. Ignored for Pinecone
- `dataset_id` (string, optional) -- Apify dataset ID from RAG Embedding Generator. Items must have an `embedding` field. Takes priority over `vectors`
- `vectors` (array, optional) -- Direct vector input as JSON array. Each item needs an `embedding` field
- `batch_size` (integer, optional) -- Vectors per upsert request. Default: 100. Pinecone max: 1000, Qdrant max: 500
- `id_field` (string, optional) -- Field to use as vector ID. Default: `"chunk_id"`. Falls back to UUID if missing

At least one of `dataset_id` or `vectors` must be provided, plus `api_key` and `index_name`.

## Usage

### Local (CLI)
```bash
APIFY_TOKEN=your-token apify run
```

### Pinecone -- Dataset Chaining
```json
{
  "api_key": "pcsk_your-pinecone-key",
  "provider": "pinecone",
  "index_name": "my-rag-index",
  "namespace": "docs",
  "dataset_id": "abc123XYZ",
  "batch_size": 200
}
```

### Qdrant -- Dataset Chaining
```json
{
  "api_key": "your-qdrant-api-key",
  "provider": "qdrant",
  "index_name": "my-collection",
  "environment": "https://xyz-example.us-east-1.aws.cloud.qdrant.io:6333",
  "distance_metric": "Cosine",
  "dataset_id": "abc123XYZ"
}
```

### Raw Vectors (Direct Input)
```json
{
  "api_key": "pcsk_your-pinecone-key",
  "provider": "pinecone",
  "index_name": "my-index",
  "vectors": [
    {
      "chunk_id": "doc1-chunk0",
      "embedding": [0.0123, -0.0456, 0.0789],
      "source_url": "https://example.com/page",
      "page_title": "Example Page"
    },
    {
      "chunk_id": "doc1-chunk1",
      "embedding": [0.0321, -0.0654, 0.0987],
      "source_url": "https://example.com/page",
      "page_title": "Example Page"
    }
  ]
}
```

### Example Output
A summary item is pushed to the default dataset:
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

This actor fills the vector storage step in a standard RAG pipeline:

```
Crawl (Website Content Crawler, 101K+ users)
  -> Clean (optional preprocessing)
    -> Chunk (RAG Content Chunker)
      -> Embed (RAG Embedding Generator)
        -> Store (this actor)
```

### Chaining with RAG Embedding Generator

1. Run RAG Content Chunker on your text or crawler output
2. Run RAG Embedding Generator with the chunker's output dataset ID
3. Copy the embedding generator's output dataset ID
4. Pass it as `dataset_id` to this actor
5. This actor reads each embedding, skips `_summary` rows, and upserts vectors with all metadata preserved

The full pipeline: crawl -> chunk -> embed -> store. Three dataset IDs passed between four actors.

## Architecture
- `src/agent/main.py` -- Actor entry point, input routing (dataset/vectors), dataset loading, provider dispatch
- `src/agent/validation.py` -- Input validation, provider whitelist, index name regex, Qdrant URL pattern check (SSRF prevention), batch size limits
- `src/agent/writers/pinecone.py` -- Pinecone writer: host resolution via control plane, batched upserts, retry, metadata mapping
- `src/agent/writers/qdrant.py` -- Qdrant writer: collection auto-creation, batched upserts, retry, payload mapping
- `src/agent/pricing.py` -- PPE billing calculator ($0.0004/vector)
- `skill.md` -- Machine-readable skill contract for agent discovery

## Security
- **API key handling**: Marked `isSecret` in input schema, validated for presence only, never logged or stored, stripped from error messages via `_sanitize_error()`
- **SSRF prevention**: Pinecone host resolved via hardcoded control plane URL (`api.pinecone.io`). Qdrant cluster URLs validated against strict `cloud.qdrant.io` pattern
- **Provider whitelist**: Only `pinecone` and `qdrant` accepted
- **Input sanitization**: Control characters stripped, index names and dataset IDs regex-validated, batch sizes bounded
- **Error safety**: All error messages pass through sanitization to prevent API key leakage
- **No data retention**: Vectors exist only in memory during the run

## Pricing
Pay-Per-Event (PPE): **$0.0004 per vector** ($0.40 per 1,000 vectors).

This is the actor's platform fee only. You also pay the vector database provider (Pinecone or Qdrant) directly via your own account.

| Content Size | Approx. Vectors | Actor Fee | Notes |
|-------------|----------------|-----------|-------|
| Single blog post | 10-20 | $0.004-$0.008 | Typical small doc |
| 10-page website | 50-100 | $0.02-$0.04 | Standard site |
| 100-page docs site | 500-1,000 | $0.20-$0.40 | Documentation portal |
| Large knowledge base | 5,000-10,000 | $2.00-$4.00 | Enterprise scale |

## Troubleshooting
- **"API key is required"**: Provide your Pinecone or Qdrant API key in the `api_key` field
- **"Invalid provider"**: Must be `"pinecone"` or `"qdrant"`
- **"Index/collection name is required"**: Provide the name of your Pinecone index or Qdrant collection
- **"Invalid Qdrant cluster URL"**: Must match `https://{cluster}.cloud.qdrant.io:6333` pattern
- **"No input provided"**: Supply either `dataset_id` or `vectors`
- **"API key is invalid or expired"**: Your provider key was rejected. Verify it in your Pinecone/Qdrant dashboard
- **"Could not resolve host for Pinecone index"**: The index name doesn't exist in your Pinecone project
- **"Failed after 3 attempts"**: Transient API error. Try again, or reduce `batch_size`
- **"No items with valid 'embedding' arrays"**: Dataset items must have an `embedding` field with a non-empty float array

## License
See `LICENSE` file for details.

---

## MCP Integration

This actor works as an MCP tool through Apify's hosted MCP server. No custom server needed.

- **Endpoint:** `https://mcp.apify.com?tools=labrat011/rag-vector-store-writer`
- **Auth:** `Authorization: Bearer <APIFY_TOKEN>`
- **Transport:** Streamable HTTP
- **Works with:** Claude Desktop, Cursor, VS Code, Windsurf, Warp, Gemini CLI

**Example MCP config (Claude Desktop / Cursor):**

```json
{
    "mcpServers": {
        "rag-vector-store-writer": {
            "url": "https://mcp.apify.com?tools=labrat011/rag-vector-store-writer",
            "headers": {
                "Authorization": "Bearer <APIFY_TOKEN>"
            }
        }
    }
}
```

AI agents can use this actor to write embedding vectors to Pinecone or Qdrant, build vector indexes, and populate RAG knowledge bases -- all as a callable MCP tool.
