"""
SGLang embedding configuration.

SGLang exposes an OpenAI-compatible /v1/embeddings endpoint.
"""

from typing import Optional

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues

from ...hosted_vllm.embedding.transformation import HostedVLLMEmbeddingConfig


class SGLangEmbeddingConfig(HostedVLLMEmbeddingConfig):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("SGLANG_API_KEY") or "fake-api-key"

        default_headers = {
            "Content-Type": "application/json",
        }
        if api_key and api_key != "fake-api-key":
            default_headers["Authorization"] = f"Bearer {api_key}"
        return {**default_headers, **headers}

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str("SGLANG_API_BASE") or "http://127.0.0.1:30000/v1"
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/embeddings"):
            api_base = f"{api_base}/embeddings"
        return api_base

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        if isinstance(input, str):
            input = [input]
        if model.startswith("sglang/"):
            model = model.replace("sglang/", "", 1)
        return {
            "model": model,
            "input": input,
            **optional_params,
        }
