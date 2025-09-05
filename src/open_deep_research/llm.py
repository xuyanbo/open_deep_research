import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Tuple


_openai_semaphore: Optional[asyncio.Semaphore] = None
_llm_limit: Optional[int] = None


def get_base_url_for_model(model: str) -> Tuple[str, Optional[str]]:
    """Return the cleaned model name and base URL for prefixed model identifiers.

    Model identifiers may optionally include a provider prefix separated by a
    colon (e.g. ``"openai:gpt-4.1"``). This function strips the prefix and
    returns the remaining model name along with the base URL that should be used
    for API requests. It now also understands the ``openai:`` prefix and will
    read the ``OPENAI_BASE_URL`` environment variable to allow routing requests
    to self-hosted or proxy OpenAI-compatible services.

    Currently supported prefixes:
    ``openai`` – base URL is read from the ``OPENAI_BASE_URL`` environment
    variable if set, otherwise defaults to the official OpenAI endpoint. This
    enables routing to custom OpenAI compatible services.

    Args:
        model: The model identifier which may contain a provider prefix.

    Returns:
        A tuple of ``(model_name, base_url)`` where ``base_url`` may be ``None``
        when no custom endpoint is required.
    """
    base_url: Optional[str] = None
    model_name = model

    if ":" in model:
        provider, model_name = model.split(":", 1)
        provider = provider.lower()

        if provider == "openai":
            # Allow overriding the OpenAI base URL to support compatible services
            base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        elif provider == "vllm":
            base_url = os.getenv("VLLM_BASE_URL")

    return model_name, base_url


def _get_llm_limit() -> Optional[int]:
    """Return the concurrency limit for OpenAI-compatible models if configured.

    The limit is only honored when ``OPENAI_BASE_URL`` is specified. If
    ``LLM_CONCURRENCY_LIMIT`` is unset or invalid the function returns
    ``None`` indicating no concurrency control.
    """
    if not os.getenv("OPENAI_BASE_URL"):
        return None

    limit_str = os.getenv("LLM_CONCURRENCY_LIMIT")
    if not limit_str:
        return None

    try:
        limit = int(limit_str)
    except ValueError:
        return None

    return max(limit, 1)


def get_openai_semaphore() -> Optional[asyncio.Semaphore]:
    """Return a semaphore enforcing the configured concurrency limit."""
    limit = _get_llm_limit()
    if limit is None:
        return None

    global _openai_semaphore, _llm_limit
    # Re-create the semaphore if the limit changed or hasn't been created
    if _openai_semaphore is None or _llm_limit != limit:
        _openai_semaphore = asyncio.Semaphore(limit)
        _llm_limit = limit

    return _openai_semaphore


@asynccontextmanager
async def openai_concurrency() -> AsyncGenerator[None, None]:
    """Async context manager that limits concurrent OpenAI requests.

    When ``OPENAI_BASE_URL`` is configured and ``LLM_CONCURRENCY_LIMIT`` is a
    positive integer, concurrent requests are gated by an ``asyncio.Semaphore``.
    Without a custom base URL no concurrency limiting occurs, preserving the
    existing behavior.
    """
    semaphore = get_openai_semaphore()
    if semaphore is None:
        yield
    else:
        async with semaphore:
            yield
