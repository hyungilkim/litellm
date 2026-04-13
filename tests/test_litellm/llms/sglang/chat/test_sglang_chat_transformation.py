import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.sglang.chat.transformation import SGLangChatConfig


def test_sglang_api_base_normalization_appends_v1():
    config = SGLangChatConfig()
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base="http://host:30000",
        api_key=None,
    )
    assert api_base == "http://host:30000/v1"
    assert api_key == "fake-api-key"


def test_sglang_api_base_normalization_keeps_existing_v1():
    config = SGLangChatConfig()
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base="http://host:30000/v1",
        api_key=None,
    )
    assert api_base == "http://host:30000/v1"
    assert api_key == "fake-api-key"


def test_sglang_env_fallback(monkeypatch):
    config = SGLangChatConfig()
    monkeypatch.setenv("SGLANG_API_BASE", "http://env-host:30000")
    monkeypatch.setenv("SGLANG_API_KEY", "env-api-key")

    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None,
        api_key=None,
    )
    assert api_base == "http://env-host:30000/v1"
    assert api_key == "env-api-key"
