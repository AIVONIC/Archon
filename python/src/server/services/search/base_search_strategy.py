"""
Base Search Strategy

Implements the foundational vector similarity search that all other strategies build upon.
This is the core semantic search functionality.
"""

from typing import Any

from ...config.logfire_config import get_logger, safe_span
from ..storage import DatabaseBackend

logger = get_logger(__name__)

# Fixed similarity threshold for vector results
SIMILARITY_THRESHOLD = 0.05


class BaseSearchStrategy:
    """Base strategy implementing fundamental vector similarity search"""

    def __init__(self, backend: DatabaseBackend):
        """Initialize with the database backend"""
        self.backend = backend

    async def vector_search(
        self,
        query_embedding: list[float],
        match_count: int,
        filter_metadata: dict | None = None,
        table_rpc: str = "match_archon_crawled_pages",
    ) -> list[dict[str, Any]]:
        """
        Perform basic vector similarity search.

        This is the foundational semantic search that all strategies use.
        Automatically detects embedding dimension and routes to the correct
        multi-dimensional RPC function.

        Args:
            query_embedding: The embedding vector for the query
            match_count: Number of results to return
            filter_metadata: Optional metadata filters
            table_rpc: The RPC function to call (match_archon_crawled_pages or match_archon_code_examples)

        Returns:
            List of matching documents with similarity scores
        """
        with safe_span("base_vector_search", table=table_rpc, match_count=match_count) as span:
            try:
                embedding_dim = len(query_embedding)

                # Use multi-dimensional RPC for non-1536 embeddings
                if embedding_dim != 1536:
                    actual_rpc = table_rpc + "_multi" if not table_rpc.endswith("_multi") else table_rpc
                    rpc_params = {
                        "query_embedding": query_embedding,
                        "embedding_dimension": embedding_dim,
                        "match_count": match_count,
                    }
                else:
                    actual_rpc = table_rpc
                    rpc_params = {"query_embedding": query_embedding, "match_count": match_count}

                # Add filter parameters
                if filter_metadata:
                    if "source" in filter_metadata:
                        rpc_params["source_filter"] = filter_metadata["source"]
                        rpc_params["filter"] = {}
                    else:
                        rpc_params["filter"] = filter_metadata
                else:
                    rpc_params["filter"] = {}

                # Execute search
                rows = await self.backend.rpc(actual_rpc, rpc_params)

                # Filter by similarity threshold
                filtered_results = []
                for result in rows:
                    similarity = float(result.get("similarity", 0.0))
                    if similarity >= SIMILARITY_THRESHOLD:
                        filtered_results.append(result)

                span.set_attribute("results_found", len(filtered_results))
                span.set_attribute("results_filtered", len(rows) - len(filtered_results))

                return filtered_results

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                span.set_attribute("error", str(e))
                return []
