"""
Contextual Embedding Service

Handles generation of contextual embeddings for improved RAG retrieval.
Includes proper rate limiting for OpenAI API calls.
"""

import os
import re

import openai

from ...config.logfire_config import search_logger
from ..credential_service import credential_service
from ..llm_provider_service import (
    extract_message_text,
    get_llm_client,
    prepare_chat_completion_params,
    requires_max_completion_tokens,
)
from ..threading_service import get_threading_service


async def generate_contextual_embedding(
    full_document: str, chunk: str, provider: str = None
) -> tuple[str, bool]:
    """
    Generate contextual information for a chunk with proper rate limiting.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        provider: Optional provider override

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    # Model choice is a RAG setting, get from credential service
    try:
        model_choice = await credential_service.get_credential("MODEL_CHOICE", "gpt-4.1-nano")
    except Exception as e:
        # Fallback to environment variable or default
        search_logger.warning(
            f"Failed to get MODEL_CHOICE from credential service: {e}, using fallback"
        )
        model_choice = os.getenv("MODEL_CHOICE", "gpt-4.1-nano")

    search_logger.debug(f"Using MODEL_CHOICE: {model_choice}")

    threading_service = get_threading_service()

    # Estimate tokens: document preview (5000 chars ≈ 1250 tokens) + chunk + prompt
    estimated_tokens = 1250 + len(chunk.split()) + 100  # Rough estimate

    try:
        # Use rate limiting before making the API call
        async with threading_service.rate_limited_operation(estimated_tokens):
            async with get_llm_client(provider=provider) as client:
                prompt = f"""<document>
{full_document[:5000]}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

                # Get model from provider configuration
                model = await _get_model_choice(provider)

                # Prepare parameters and convert max_tokens for GPT-5/reasoning models
                params = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that provides concise contextual information.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1200 if requires_max_completion_tokens(model) else 200,  # Much more tokens for reasoning models (GPT-5 needs extra for reasoning process)
                }
                final_params = prepare_chat_completion_params(model, params)
                response = await client.chat.completions.create(**final_params)

                choice = response.choices[0] if response.choices else None
                context, _, _ = extract_message_text(choice)
                context = context.strip()
                contextual_text = f"{context}\n---\n{chunk}"

                return contextual_text, True

    except Exception as e:
        if "rate_limit_exceeded" in str(e) or "429" in str(e):
            search_logger.warning(f"Rate limit hit in contextual embedding: {e}")
        else:
            search_logger.error(f"Error generating contextual embedding: {e}")
        return chunk, False


async def process_chunk_with_context(
    url: str, content: str, full_document: str
) -> tuple[str, bool]:
    """
    Process a single chunk with contextual embedding using async/await.

    Args:
        url: URL of the document
        content: The chunk content
        full_document: The complete document text

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    return await generate_contextual_embedding(full_document, content)


async def _get_model_choice(provider: str | None = None) -> str:
    """Get model choice from credential service with centralized defaults."""
    from ..credential_service import credential_service

    # Get the active provider configuration
    provider_config = await credential_service.get_active_provider("llm")
    model = provider_config.get("chat_model", "").strip()  # Strip whitespace
    provider_name = provider_config.get("provider", "openai")

    # Handle empty model case - use centralized defaults
    if not model:
        search_logger.warning(f"chat_model is empty for provider {provider_name}, using centralized defaults")

        # Special handling for Ollama to check specific credential
        if provider_name == "ollama":
            try:
                ollama_model = await credential_service.get_credential("OLLAMA_CHAT_MODEL")
                if ollama_model and ollama_model.strip():
                    model = ollama_model.strip()
                    search_logger.info(f"Using OLLAMA_CHAT_MODEL fallback: {model}")
                else:
                    # Use default for Ollama
                    model = "llama3.2:latest"
                    search_logger.info(f"Using Ollama default: {model}")
            except Exception as e:
                search_logger.error(f"Error getting OLLAMA_CHAT_MODEL: {e}")
                model = "llama3.2:latest"
                search_logger.info(f"Using Ollama fallback: {model}")
        else:
            # Use provider-specific defaults
            provider_defaults = {
                "openai": "gpt-4o-mini",
                "openrouter": "anthropic/claude-3.5-sonnet",
                "google": "gemini-1.5-flash",
                "anthropic": "claude-3-5-haiku-20241022",
                "grok": "grok-3-mini"
            }
            model = provider_defaults.get(provider_name, "gpt-4o-mini")
            search_logger.debug(f"Using default model for provider {provider_name}: {model}")
    search_logger.debug(f"Using model from credential service: {model}")

    return model


# A contextual call must fit the chat model's context window per request (16,384 tokens per slot on the
# DGX Qwen lane). Text full of URLs packs about 1.9 characters per token (37,500 characters measured as
# 19,688 tokens, 2026-11-05), so this budget stays near 8,500 prompt tokens even for that kind of text,
# leaving room for the answer.
CONTEXTUAL_PROMPT_MAX_CHARS = int(os.getenv("CONTEXTUAL_PROMPT_MAX_CHARS", "16000"))
DOCUMENT_PREVIEW_CHARS = 2000
CHUNK_PREVIEW_CHARS = 500
_CHUNK_LINE = re.compile(r"^\s*\**\s*CHUNK\s+(\d+)\s*\**\s*:\s*\**\s*(.+)$", re.IGNORECASE)


class ContextTooLarge(Exception):
    """The model refused the request because the prompt exceeds its context window."""


def _is_context_overflow(error: Exception) -> bool:
    text = str(error)
    # llama.cpp: exceed_context_size_error; OpenAI: context_length_exceeded
    return "exceed_context_size" in text or "context_length_exceeded" in text


def pack_contextual_calls(previews: list[str], chunks: list[str], max_chars: int) -> list[list[int]]:
    """Group chunk indexes into calls whose prompt stays within max_chars.

    Each distinct document preview is counted once per call, because the prompt carries it once.
    A single chunk always gets a call of its own, even when it alone is over the budget.
    """
    calls: list[list[int]] = []
    current: list[int] = []
    seen: set[str] = set()
    size = 0
    for i, chunk in enumerate(chunks):
        cost = len(chunk) + (0 if previews[i] in seen else len(previews[i]))
        if current and size + cost > max_chars:
            calls.append(current)
            current, seen, size = [], set(), 0
            cost = len(chunk) + len(previews[i])
        current.append(i)
        seen.add(previews[i])
        size += cost
    if current:
        calls.append(current)
    return calls


def build_contextual_prompt(previews: list[str], chunks: list[str]) -> str:
    """One prompt for several chunks. Each document preview appears once; chunks refer to it by number."""
    doc_numbers: dict[str, int] = {}
    for preview in previews:
        doc_numbers.setdefault(preview, len(doc_numbers) + 1)
    prompt = "Process the following chunks and provide contextual information for each:\n\n"
    for preview, number in doc_numbers.items():
        prompt += f'<document id="{number}">\n{preview}\n</document>\n\n'
    for i, (preview, chunk) in enumerate(zip(previews, chunks, strict=True)):
        prompt += f'CHUNK {i + 1}:\n<chunk document="{doc_numbers[preview]}">\n{chunk}\n</chunk>\n\n'
    prompt += (
        "For each chunk, provide a short succinct context to situate it within its document for improving "
        "search retrieval. Format your response as:\nCHUNK 1: [context]\nCHUNK 2: [context]\netc."
    )
    return prompt


def parse_contextual_response(response_text: str) -> dict[int, str]:
    """Map zero-based chunk index to its context from lines like 'CHUNK 3: ...'."""
    contexts: dict[int, str] = {}
    for line in response_text.strip().split("\n"):
        match = _CHUNK_LINE.match(line)
        if match and match.group(2).strip():
            contexts[int(match.group(1)) - 1] = match.group(2).strip()
    return contexts


async def _contextual_call(client, model_choice: str, previews: list[str], chunks: list[str]) -> list[tuple[str, bool]]:
    """One model call for these chunks. Raises ContextTooLarge when the prompt does not fit."""
    batch_params = {
        "model": model_choice,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates contextual information for document chunks.",
            },
            {"role": "user", "content": build_contextual_prompt(previews, [c[:CHUNK_PREVIEW_CHARS] for c in chunks])},
        ],
        "temperature": 0,
        # Reasoning models (GPT-5) need room for their reasoning as well as the answer
        "max_tokens": (600 if requires_max_completion_tokens(model_choice) else 100) * len(chunks),
    }
    try:
        response = await client.chat.completions.create(**prepare_chat_completion_params(model_choice, batch_params))
    except openai.BadRequestError as e:
        if _is_context_overflow(e):
            raise ContextTooLarge(str(e)) from e
        raise

    choice = response.choices[0] if response.choices else None
    response_text, _, _ = extract_message_text(choice)
    if not response_text:
        search_logger.error(f"Empty response from LLM when generating contextual embeddings for {len(chunks)} chunks")
        return [(chunk, False) for chunk in chunks]

    contexts = parse_contextual_response(response_text)
    # The context goes in front of the full chunk, not the preview the model saw
    return [
        (contexts[i] + "\n\n" + chunk, True) if i in contexts else (chunk, False)
        for i, chunk in enumerate(chunks)
    ]


async def _contextual_call_splitting(client, model_choice: str, previews: list[str], chunks: list[str]) -> list[tuple[str, bool]]:
    """Run one call; when the model says the prompt is too large, halve it and try each half."""
    try:
        return await _contextual_call(client, model_choice, previews, chunks)
    except ContextTooLarge as e:
        if len(chunks) == 1:
            search_logger.error(f"Contextual embedding skipped: one chunk with its document preview exceeds the model context: {e}")
            return [(chunks[0], False)]
        half = len(chunks) // 2
        search_logger.warning(f"Contextual prompt for {len(chunks)} chunks exceeds the model context; splitting into {half} + {len(chunks) - half}")
        return (await _contextual_call_splitting(client, model_choice, previews[:half], chunks[:half])
                + await _contextual_call_splitting(client, model_choice, previews[half:], chunks[half:]))


async def generate_contextual_embeddings_batch(
    full_documents: list[str], chunks: list[str], provider: str = None
) -> list[tuple[str, bool]]:
    """
    Generate contextual information for multiple chunks with as few model calls as fit.

    The chunks are packed into calls by prompt size (CONTEXTUAL_PROMPT_MAX_CHARS), with each document's
    preview sent once per call. A call the model still refuses as too large is split in half and retried.

    Args:
        full_documents: List of complete document texts, one per chunk
        chunks: List of specific chunks to generate context for
        provider: Optional provider override

    Returns:
        List of tuples, in the order of chunks, containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    previews = [doc[:DOCUMENT_PREVIEW_CHARS] for doc in full_documents]
    sizes = [min(len(c), CHUNK_PREVIEW_CHARS) for c in chunks]
    results: list[tuple[str, bool]] = [(chunk, False) for chunk in chunks]
    calls = pack_contextual_calls(previews, ["x" * n for n in sizes], CONTEXTUAL_PROMPT_MAX_CHARS)
    try:
        async with get_llm_client(provider=provider) as client:
            model_choice = await _get_model_choice(provider)
            for call in calls:
                try:
                    call_results = await _contextual_call_splitting(
                        client, model_choice, [previews[i] for i in call], [chunks[i] for i in call]
                    )
                except openai.RateLimitError as e:
                    if "insufficient_quota" in str(e):
                        search_logger.warning(f"Quota exhausted in contextual embeddings, proceeding without them: {e}")
                    else:
                        search_logger.warning(f"Rate limit hit in contextual embeddings, proceeding without them for this call: {e}")
                    continue
                except Exception as e:
                    search_logger.error(f"Error in contextual embedding call for {len(call)} chunks: {e}", exc_info=True)
                    continue
                for i, result in zip(call, call_results, strict=True):
                    results[i] = result
    except Exception as e:
        search_logger.error(f"Error in contextual embedding batch: {e}", exc_info=True)
    return results
