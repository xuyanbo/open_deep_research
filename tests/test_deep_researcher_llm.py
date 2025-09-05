import asyncio
from contextlib import asynccontextmanager

import pytest


def test_clarify_with_user_uses_custom_base_url_and_concurrency(monkeypatch):
    pytest.importorskip("langchain")
    pytest.importorskip("langgraph")
    HumanMessage = pytest.importorskip("langchain_core.messages").HumanMessage
    RunnableConfig = pytest.importorskip("langchain_core.runnables").RunnableConfig
    deep_researcher = pytest.importorskip("open_deep_research.deep_researcher")
    ClarifyWithUser = pytest.importorskip("open_deep_research.state").ClarifyWithUser

    custom_url = "https://example.com/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", custom_url)
    monkeypatch.setenv("LLM_CONCURRENCY_LIMIT", "1")

    calls: list[bool] = []

    @asynccontextmanager
    async def fake_concurrency():
        calls.append(True)
        yield

    monkeypatch.setattr(deep_researcher, "openai_concurrency", fake_concurrency)

    class DummyModel:
        def with_structured_output(self, *args, **kwargs):
            return self

        def with_retry(self, *args, **kwargs):
            return self

        def with_config(self, config):
            self.config = config
            return self

        async def ainvoke(self, messages):
            return ClarifyWithUser(need_clarification=False, question="", verification="ok")

    dummy = DummyModel()
    monkeypatch.setattr(deep_researcher, "configurable_model", dummy)

    state = {"messages": [HumanMessage(content="hi")]}  # minimal AgentState
    config = RunnableConfig()

    asyncio.run(deep_researcher.clarify_with_user(state, config))

    assert dummy.config["base_url"] == custom_url
    assert calls, "openai_concurrency should be used"
