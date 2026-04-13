from typing import List, Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class SGLangChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> List[str]:
        return super().get_supported_openai_params(model)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

    @staticmethod
    def _normalize_api_base(api_base: Optional[str]) -> Optional[str]:
        if api_base is None:
            return None
        normalized = api_base.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized
        return f"{normalized}/v1"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("SGLANG_API_BASE")  # type: ignore
        api_base = self._normalize_api_base(api_base)
        dynamic_api_key = api_key or get_secret_str("SGLANG_API_KEY") or "fake-api-key"
        return api_base, dynamic_api_key
