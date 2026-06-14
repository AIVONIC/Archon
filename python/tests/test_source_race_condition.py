"""
Test race condition handling in source creation.

This test ensures that concurrent source creation attempts use UPSERT (never
INSERT) so they don't fail with PRIMARY KEY violations.
"""

import asyncio


from src.server.services.source_management_service import update_source_info
from tests.fake_backend import FakeBackend


class TestSourceRaceCondition:
    """Test that concurrent source creation handles race conditions properly."""

    async def test_concurrent_source_creation_no_race(self):
        """Concurrent attempts to create the same source should all succeed."""
        # No existing source: every attempt takes the new-source upsert path.
        backend = FakeBackend(data=[])

        async def create_source(thread_id):
            await update_source_info(
                backend=backend,
                source_id="test_source_123",
                summary=f"Summary from thread {thread_id}",
                word_count=100,
                content=f"Content from thread {thread_id}",
                knowledge_type="documentation",
                tags=["test"],
                update_frequency=0,
                source_url="https://example.com",
                source_display_name=f"Example Site {thread_id}",
            )

        await asyncio.gather(*(create_source(i) for i in range(5)))

        upserts = [c for c in backend.calls if c[0] == "upsert"]
        assert len(upserts) == 5, "All 5 attempts should upsert"
        assert not any(c[0] == "insert" for c in backend.calls), "Must never insert"

    async def test_upsert_vs_insert_behavior(self):
        """A brand new source should be created via UPSERT, not INSERT."""
        backend = FakeBackend(data=[])  # source does not exist

        await update_source_info(
            backend=backend,
            source_id="new_source",
            summary="Test summary",
            word_count=100,
            content="Test content",
            knowledge_type="documentation",
            source_display_name="Test Display Name",
        )

        methods = [c[0] for c in backend.calls]
        assert "upsert" in methods, "Should use upsert for new sources"
        assert "insert" not in methods, "Should not use insert to avoid race conditions"

    async def test_existing_source_uses_upsert(self):
        """An existing source should also be updated via UPSERT (not UPDATE)."""
        existing_source = {
            "source_id": "existing_source",
            "title": "Existing Title",
            "metadata": {"knowledge_type": "api"},
        }
        backend = FakeBackend(data=[existing_source])  # source exists

        await update_source_info(
            backend=backend,
            source_id="existing_source",
            summary="Updated summary",
            word_count=200,
            content="Updated content",
            knowledge_type="documentation",
        )

        methods = [c[0] for c in backend.calls]
        assert "upsert" in methods, "Should use upsert for existing sources"
        assert "update" not in methods, "Should not use update (upsert handles races)"

    async def test_async_concurrent_creation(self):
        """Concurrent async creation of two sources should all upsert."""
        backend = FakeBackend(data=[])

        async def create_source_async(task_id):
            await update_source_info(
                backend=backend,
                source_id=f"async_source_{task_id % 2}",  # Only 2 unique sources
                summary=f"Summary {task_id}",
                word_count=100,
                content=f"Content {task_id}",
                knowledge_type="documentation",
            )

        await asyncio.gather(*(create_source_async(i) for i in range(10)))

        upserts = [c for c in backend.calls if c[0] == "upsert"]
        assert len(upserts) == 10, "All 10 operations should upsert"

    async def test_race_condition_resolved_by_upsert(self):
        """Even if a concurrent writer wins between check and write, upsert is safe."""
        # Existing data means the check sees the source; the write still upserts.
        backend = FakeBackend(data=[{"source_id": "race_source", "title": "Existing", "metadata": {}}])

        async def create(thread_id):
            await update_source_info(
                backend=backend,
                source_id="race_source",
                summary="Race summary",
                word_count=100,
                content="Race content",
                knowledge_type="documentation",
                source_display_name="Race Display Name",
            )

        await asyncio.gather(create(1), create(2))

        assert all(c[0] != "insert" for c in backend.calls), "Race is resolved with upsert, never insert"
        assert any(c[0] == "upsert" for c in backend.calls)
