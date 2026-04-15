"""
Tests for SGLang provider configuration and integration.
"""

import os
import sys

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestSGLangProviderConfig:
    """Test SGLang provider configuration."""

    def test_sglang_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "SGLANG")
        assert LlmProviders.SGLANG.value == "sglang"
        assert "sglang" in litellm.provider_list

    def test_sglang_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="sglang/Qwen/Qwen3-4B",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "Qwen/Qwen3-4B"
        assert provider == "sglang"
        assert api_base == "http://127.0.0.1:30000/v1"
        assert api_key == "fake-api-key"

    def test_sglang_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "qwen3-4b",
                    "litellm_params": {
                        "model": "sglang/Qwen/Qwen3-4B",
                        "api_base": "http://127.0.0.1:30000/v1",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "qwen3-4b"

    def test_sglang_embedding_config(self):
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager
        from litellm.llms.sglang.embedding.transformation import SGLangEmbeddingConfig

        config = ProviderConfigManager.get_provider_embedding_config(
            model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            provider=LlmProviders.SGLANG,
        )

        assert isinstance(config, SGLangEmbeddingConfig)

    def test_sglang_model_info_reuses_vllm_model_info(self):
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        model_info = ProviderConfigManager.get_provider_model_info(
            model="Qwen/Qwen3-4B", provider=LlmProviders.SGLANG
        )

        assert model_info is not None
