"""Contextual embedding batches must fit the chat model's context window.

On 2026-11-05 every 15-chunk batch of a product sitemap was refused by the DGX Qwen lane
("request (19688 tokens) exceeds the available context size (16384 tokens)"), because the
prompt repeated the same 2,000-character document preview once per chunk, and the whole batch
was stored without context. These tests drive generate_contextual_embeddings_batch against a
fake model that refuses an oversized prompt the way llama.cpp does.
"""

import re
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.server.services.embeddings import contextual_embedding_service as ces


class FakeModel:
    """Answers 'CHUNK n: context for n' for every chunk, or refuses a prompt over max_prompt_chars."""

    def __init__(self, max_prompt_chars: int = 10**9, fail_calls: set[int] | None = None):
        self.max_prompt_chars = max_prompt_chars
        self.fail_calls = fail_calls or set()
        self.prompts: list[str] = []
        self.events: list[str] = []
        self.base_url = "http://dgx/v1/"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **params):
        prompt = params["messages"][1]["content"]
        self.prompts.append(prompt)
        self.events.append("call")
        if len(self.prompts) - 1 in self.fail_calls:
            raise RuntimeError("model unavailable")
        if len(prompt) > self.max_prompt_chars:
            request = httpx.Request("POST", "http://dgx/v1/chat/completions")
            raise openai.BadRequestError(
                "Error code: 400 - {'error': {'code': 400, 'message': 'request (19688 tokens) exceeds the "
                "available context size (16384 tokens), try increasing it', 'type': 'exceed_context_size_error'}}",
                response=httpx.Response(400, request=request),
                body=None,
            )
        n = len(re.findall(r"^CHUNK \d+:$", prompt, re.MULTILINE))
        text = "\n".join(f"CHUNK {i + 1}: context for {i + 1}" for i in range(n))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture
def model(monkeypatch):
    holder = {"model": FakeModel()}

    @asynccontextmanager
    async def fake_client(provider=None):
        yield holder["model"]

    async def fake_model_choice(provider=None):
        return "qwen"

    monkeypatch.setattr(ces, "get_llm_client", fake_client)
    monkeypatch.setattr(ces, "_get_model_choice", fake_model_choice)
    monkeypatch.setattr(ces, "extract_message_text", lambda choice: (choice.message.content, None, None))
    # These tests are about packing by prompt size; the chunk cap and the lane wait have their own tests.
    monkeypatch.setattr(ces, "CONTEXTUAL_MAX_CHUNKS_PER_CALL", None)

    async def record_wait(self):
        holder["model"].events.append("wait")
        return 0.0

    monkeypatch.setattr(ces.ChatLane, "wait_until_quiet", record_wait)
    return holder


def _sitemap(n_chunks: int) -> tuple[list[str], list[str]]:
    doc = "".join(f"https://shop.example.se/products/item-{i}\n" for i in range(3000))
    chunks = [doc[i * 600:(i + 1) * 600] for i in range(n_chunks)]
    return [doc] * n_chunks, chunks


@pytest.mark.asyncio
async def test_fifteen_chunks_of_one_document_send_its_preview_once(model):
    docs, chunks = _sitemap(15)
    results = await ces.generate_contextual_embeddings_batch(docs, chunks)

    assert all(ok for _, ok in results)
    assert len(model["model"].prompts) == 1
    prompt = model["model"].prompts[0]
    assert prompt.count(docs[0][:ces.DOCUMENT_PREVIEW_CHARS]) == 1
    assert len(prompt) <= ces.CONTEXTUAL_PROMPT_MAX_CHARS + 1000


@pytest.mark.asyncio
async def test_a_prompt_the_model_refuses_is_split_until_it_fits(model):
    docs, chunks = _sitemap(15)
    model["model"] = FakeModel(max_prompt_chars=5000)
    results = await ces.generate_contextual_embeddings_batch(docs, chunks)

    assert all(ok for _, ok in results)
    assert len(model["model"].prompts) > 1
    for text, _ in results:
        assert text.startswith("context for ")


@pytest.mark.asyncio
async def test_distinct_documents_are_packed_within_the_budget(model, monkeypatch):
    monkeypatch.setattr(ces, "CONTEXTUAL_PROMPT_MAX_CHARS", 6000)
    docs = [chr(65 + i) * 3000 for i in range(6)]
    chunks = [f"chunk {i} " * 40 for i in range(6)]
    results = await ces.generate_contextual_embeddings_batch(docs, chunks)

    assert all(ok for _, ok in results)
    assert len(model["model"].prompts) == 3
    for prompt in model["model"].prompts:
        assert len(prompt) < 6000 + 1000


@pytest.mark.asyncio
async def test_context_is_joined_to_the_full_chunk_with_real_newlines(model):
    chunk = "x" * 900
    results = await ces.generate_contextual_embeddings_batch(["doc"], [chunk])

    assert results == [("context for 1\n\n" + chunk, True)]


@pytest.mark.asyncio
async def test_a_failed_call_fails_only_its_own_chunks_and_order_is_kept(model, monkeypatch):
    monkeypatch.setattr(ces, "CONTEXTUAL_PROMPT_MAX_CHARS", 3000)
    model["model"] = FakeModel(fail_calls={1})
    docs = [chr(65 + i) * 2000 for i in range(3)]
    chunks = [f"chunk-{i}" for i in range(3)]
    results = await ces.generate_contextual_embeddings_batch(docs, chunks)

    assert [ok for _, ok in results] == [True, False, True]
    assert results[1] == ("chunk-1", False)
    assert results[0][0].endswith("chunk-0") and results[2][0].endswith("chunk-2")


def test_response_lines_are_parsed_with_markdown_and_spacing():
    parsed = ces.parse_contextual_response("Here you go\n**CHUNK 1:** first\n  chunk 2 : second\nCHUNK 3:")
    assert parsed == {0: "first", 1: "second"}


@pytest.mark.asyncio
async def test_calls_are_capped_and_each_waits_for_the_lane_first(model, monkeypatch):
    monkeypatch.setattr(ces, "CONTEXTUAL_MAX_CHUNKS_PER_CALL", 3)
    docs, chunks = _sitemap(15)
    results = await ces.generate_contextual_embeddings_batch(docs, chunks)

    assert all(ok for _, ok in results)
    assert [r.split("\n\n", 1)[1] for r, _ in results] == chunks
    prompts = model["model"].prompts
    assert len(prompts) == 5
    assert all(len(re.findall(r"^CHUNK \d+:$", p, re.MULTILINE)) == 3 for p in prompts)
    assert model["model"].events == ["wait", "call"] * 5


def _lane(answers: list, quiet: float, max_wait: float, monkeypatch) -> tuple[ces.ChatLane, list[str]]:
    """A ChatLane over a fake /slots that returns the answers in turn (the last one repeats)."""
    monkeypatch.setattr(ces, "CONTEXTUAL_YIELD_QUIET_S", quiet)
    monkeypatch.setattr(ces, "CONTEXTUAL_YIELD_MAX_WAIT_S", max_wait)
    monkeypatch.setattr(ces, "CONTEXTUAL_YIELD_POLL_S", 0.02)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        answer = answers[min(len(seen) - 1, len(answers) - 1)]
        if isinstance(answer, int):
            return httpx.Response(answer, text="Not Found")
        return httpx.Response(200, json=[{"id": i, "is_processing": busy} for i, busy in enumerate(answer)])

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ces.ChatLane("http://172.17.0.1:18085/v1/", http), seen


def test_slots_url_is_the_server_root_not_the_openai_path():
    lane = ces.ChatLane("http://172.17.0.1:18085/v1/", httpx.AsyncClient())
    assert lane.slots_url == "http://172.17.0.1:18085/slots"


@pytest.mark.asyncio
async def test_lane_waits_while_a_customer_request_runs_then_for_the_quiet_period(monkeypatch):
    busy, idle = [False, True, False], [False, False, False]
    lane, seen = _lane([busy] * 5 + [idle], quiet=0.1, max_wait=5, monkeypatch=monkeypatch)
    waited = await lane.wait_until_quiet()

    assert len(seen) >= 5 + 3  # five busy readings, then idle long enough to count as quiet
    assert waited >= 0.1
    assert waited < 5
    assert seen[0] == "http://172.17.0.1:18085/slots"


@pytest.mark.asyncio
async def test_a_short_gap_between_two_chat_calls_does_not_count_as_quiet(monkeypatch):
    # busy, one idle reading (the gap between a turn's routing call and its reply), busy again, then idle
    answers = [[True], [False]] + [[True]] * 15 + [[False]]
    lane, seen = _lane(answers, quiet=0.1, max_wait=5, monkeypatch=monkeypatch)
    await lane.wait_until_quiet()
    assert len(seen) > len(answers)  # it only stops once quiet after the LAST busy reading


@pytest.mark.asyncio
async def test_lane_gives_up_waiting_after_the_maximum(monkeypatch):
    lane, _ = _lane([[True]], quiet=0.05, max_wait=0.3, monkeypatch=monkeypatch)
    waited = await lane.wait_until_quiet()
    assert 0.3 <= waited < 1.0


@pytest.mark.asyncio
async def test_a_server_without_slots_is_not_waited_on_and_not_asked_again(monkeypatch):
    lane, seen = _lane([404], quiet=0.1, max_wait=5, monkeypatch=monkeypatch)
    assert await lane.wait_until_quiet() < 0.1
    assert await lane.wait_until_quiet() < 0.1
    assert len(seen) == 1
    assert lane.readable is False
