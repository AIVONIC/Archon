"""
In-flight document ingests, keyed by the source_id they are writing.

⛔ WHY THIS EXISTS. Deleting a source used to leave the upload that was still writing it
running. Workspace replaces a changed document by deleting the old one and uploading the new
one, and an ingest of a large file (LEDGER.md: 400+ chunks, one LLM summary each) takes longer
than the gap between two E3 memo pushes. So every sync deleted a source whose upload was mid-way,
and that upload went on paying for contextual summaries on the PRIMARY CHAT model and inserting
chunks for a parent row that no longer existed: 818 `archon_crawled_pages_source_id_fkey`
violations in 24h on 2026-09-26, and "Document upload completed successfully" for every one of
the abandoned copies.

Deleting a source now cancels the task writing it, BEFORE the row is deleted, so no insert can
land after the delete. The source_id is generated before its row exists, so this also stops an
upload that is still in its summary phase and would otherwise CREATE the row after the delete
and leave an orphan nobody owns.

In-process on purpose: archon-server runs a single uvicorn worker. The storage layer ALSO
re-checks that the source row exists before every batch (document_storage_service), which is
what covers a second process or a restart losing this map.
"""

import asyncio

from ...config.logfire_config import get_logger

logger = get_logger(__name__)

# source_id -> the asyncio task ingesting it
_active: dict[str, asyncio.Task] = {}


def register(source_id: str, task: asyncio.Task) -> None:
    _active[source_id] = task


def unregister(source_id: str, task: asyncio.Task | None = None) -> None:
    # Only remove our own entry: never drop a registration a later task made.
    if task is None or _active.get(source_id) is task:
        _active.pop(source_id, None)


async def cancel_ingest(source_id: str, timeout: float = 10.0) -> bool:
    """Cancel the in-flight ingest writing `source_id`. True if one was running.

    Waits (bounded) for the task to finish unwinding, so the caller's DELETE runs after the
    last insert the task could make rather than racing it.
    """
    task = _active.get(source_id)
    if task is None or task.done():
        return False
    if task is asyncio.current_task():
        return False
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except (asyncio.CancelledError, TimeoutError, asyncio.TimeoutError):
        pass
    except Exception as e:  # noqa: BLE001 - the task's own failure is not ours to raise
        logger.debug(f"cancelled ingest for {source_id} ended with {type(e).__name__}: {e}")
    logger.warning(
        f"Cancelled in-flight ingest of {source_id}: its source was deleted while it was still "
        f"being written (superseded or removed). Nothing further will be stored for it."
    )
    return True
