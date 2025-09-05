import os
import sys
import asyncio
import time
from pathlib import Path

# Ensure the src directory is on the path so we import the local package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_deep_research.llm import get_base_url_for_model, openai_concurrency


def test_openai_base_url_from_env(monkeypatch):
    custom_url = "https://example.com/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", custom_url)

    model_name, base_url = get_base_url_for_model("openai:gpt-test")

    assert model_name == "gpt-test"
    assert base_url == custom_url


def test_openai_base_url_defaults_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    model_name, base_url = get_base_url_for_model("openai:gpt-test")

    assert model_name == "gpt-test"
    assert base_url == "https://api.openai.com/v1"


def test_openai_concurrency_limit(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_CONCURRENCY_LIMIT", "1")

    async def run_tasks():
        async def worker():
            async with openai_concurrency():
                await asyncio.sleep(0.1)

        start = time.perf_counter()
        await asyncio.gather(worker(), worker())
        return time.perf_counter() - start

    elapsed = asyncio.run(run_tasks())
    assert elapsed >= 0.2


def test_openai_concurrency_disabled_without_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_CONCURRENCY_LIMIT", "1")

    async def run_tasks():
        async def worker():
            async with openai_concurrency():
                await asyncio.sleep(0.1)

        start = time.perf_counter()
        await asyncio.gather(worker(), worker())
        return time.perf_counter() - start

    elapsed = asyncio.run(run_tasks())
    assert elapsed < 0.2
