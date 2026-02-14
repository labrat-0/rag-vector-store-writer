"""
Pricing utility for RAG Vector Store Writer.

Uses Pay-Per-Event (PPE) model: charges per vector upserted.
This aligns with Apify's recommended pricing for AI/MCP-compatible actors.

Rate: $0.0004 per vector upserted (i.e., $0.40 per 1,000 vectors).
- A typical 10-page website produces ~50-100 embeddings/vectors.
- Cost for that: $0.02-$0.04.
- Note: This is the actor's fee only. The user also pays the vector DB
  provider (Pinecone, Qdrant) via their own account/API key.
"""

PER_VECTOR_RATE = 0.0004  # $0.40 per 1,000 vectors


def calculate_billing(total_vectors: int) -> dict:
    """Calculate billing based on number of vectors upserted.

    Returns:
        dict: {
            'total_vectors': int,
            'amount': float,
            'rate_per_vector': float,
        }
    """
    amount = round(total_vectors * PER_VECTOR_RATE, 6)
    return {
        "total_vectors": total_vectors,
        "amount": amount,
        "rate_per_vector": PER_VECTOR_RATE,
    }
