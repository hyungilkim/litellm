"""
SGLang chat completion configuration.

SGLang serves an OpenAI-compatible API and commonly runs locally on
http://127.0.0.1:30000/v1.
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...hosted_vllm.chat.transformation import HostedVLLMChatConfig


class SGLangChatConfig(HostedVLLMChatConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("SGLANG_API_BASE") or "http://127.0.0.1:30000/v1"
        dynamic_api_key = api_key or get_secret_str("SGLANG_API_KEY") or "fake-api-key"
        return api_base, dynamic_api_key
