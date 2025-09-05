import os
import sys
from pathlib import Path

# Ensure the src directory is on the path so we import the local package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_deep_research.llm import get_base_url_for_model


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
