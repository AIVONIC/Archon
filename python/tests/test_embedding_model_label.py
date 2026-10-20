"""The stored `embedding_model` must name the model that PRODUCED the vector.

Archon used to write the configured EMBEDDING_MODEL setting into every row. When
the embedding server served a different model than the one asked for, the label
became a false statement about the vector: on 2026-10-20 the shared VPS Archon had
10,014 chunks labelled intfloat/multilingual-e5-small whose vectors re-embedded at
cosine 1.0 with BAAI/bge-small-en-v1.5. A wrong vector does not look wrong, so the
label is the only record of which space a stored vector lives in.

These tests drive the production functions (create_embeddings_batch and
add_documents_to_supabase), with only the network and database replaced.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.server.services.embeddings.embedding_service import (
    EmbeddingBatchResult,
    create_embeddings_batch,
)
from src.server.services.storage.document_storage_service import add_documents_to_supabase
from tests.fake_backend import FakeBackend

CONFIGURED = "intfloat/multilingual-e5-small"
SERVED = "BAAI/bge-small-en-v1.5"


class _Ctx:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _client(n: int, served_model):
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.01 * (i + 1)] * 384) for i in range(n)]
    response.model = served_model
    client = MagicMock()
    client.embeddings.create = AsyncMock(return_value=response)
    return client


async def _batch(texts, served_model):
    threading = MagicMock()
    threading.rate_limited_operation.return_value = _Ctx(None)
    mod = "src.server.services.embeddings.embedding_service"
    with patch(f"{mod}.get_threading_service", return_value=threading), \
         patch(f"{mod}.get_llm_client", return_value=_Ctx(_client(len(texts), served_model))), \
         patch(f"{mod}.get_embedding_model", AsyncMock(return_value=CONFIGURED)), \
         patch(f"{mod}.credential_service") as cred:
        cred.get_active_provider = AsyncMock(return_value={"provider": "ollama"})
        cred.get_credentials_by_category = AsyncMock(
            return_value={"EMBEDDING_BATCH_SIZE": "10", "EMBEDDING_DIMENSIONS": "384"}
        )
        return await create_embeddings_batch(texts)


@pytest.mark.asyncio
async def test_batch_records_the_model_the_server_reported_not_the_setting():
    result = await _batch(["a", "b", "c"], SERVED)
    assert result.success_count == 3
    assert result.models == [SERVED, SERVED, SERVED]


@pytest.mark.asyncio
async def test_batch_falls_back_to_the_requested_model_when_none_is_reported():
    # A provider that returns no model id (here: a non-string attribute).
    result = await _batch(["a", "b"], MagicMock())
    assert result.models == [CONFIGURED, CONFIGURED]
    result = await _batch(["a"], "")
    assert result.models == [CONFIGURED]


@pytest.mark.asyncio
async def test_batch_with_honest_server_records_the_requested_model():
    result = await _batch(["a"], CONFIGURED)
    assert result.models == [CONFIGURED]


def test_model_at_falls_back_only_when_unknown():
    r = EmbeddingBatchResult()
    r.add_success([0.1] * 384, "x", SERVED)
    r.add_success([0.2] * 384, "y")  # legacy caller, no model
    assert r.model_at(0, CONFIGURED) == SERVED
    assert r.model_at(1, CONFIGURED) == CONFIGURED
    assert r.model_at(5, CONFIGURED) == CONFIGURED


@pytest.mark.asyncio
async def test_stored_row_label_is_the_producing_model():
    """End to end through the storage writer: the upserted row names SERVED."""
    contents = ["first chunk text", "second chunk text"]
    result = EmbeddingBatchResult()
    for i, t in enumerate(contents):
        result.add_success([0.1 * (i + 1)] * 384, t, SERVED)

    backend = FakeBackend(data=[{"source_id": "src1"}])  # the source exists
    mod = "src.server.services.storage.document_storage_service"
    with patch(f"{mod}.create_embeddings_batch", AsyncMock(return_value=result)), \
         patch("src.server.services.llm_provider_service.get_embedding_model",
               AsyncMock(return_value=CONFIGURED)), \
         patch(f"{mod}.credential_service") as cred:
        cred.get_credential = AsyncMock(return_value="false")
        cred.get_credentials_by_category = AsyncMock(
            return_value={"DOCUMENT_STORAGE_BATCH_SIZE": "10", "USE_CONTEXTUAL_EMBEDDINGS": "false"}
        )
        await add_documents_to_supabase(
            backend=backend,
            urls=["https://example.com/p", "https://example.com/p"],
            chunk_numbers=[0, 1],
            contents=contents,
            metadatas=[{"source_id": "src1"}, {"source_id": "src1"}],
            url_to_full_document={"https://example.com/p": "full"},
            batch_size=10,
        )

    upserts = [args[0] for name, args, _ in backend.calls if name == "upsert"]
    rows = [row for batch in upserts for row in batch]
    assert len(rows) == 2, backend.calls
    assert {row["embedding_model"] for row in rows} == {SERVED}
    assert all(row["embedding_384"] for row in rows)
