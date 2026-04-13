import os

import httpx
import pytest

import litellm


def _is_sglang_ready(base_url: str) -> bool:
    for health_path in ("/openapi.json", "/docs"):
        try:
            response = httpx.get(
                f"{base_url.rstrip('/')}{health_path}",
                timeout=5.0,
            )
            if response.status_code == 200:
                return True
        except Exception:
            pass
    return False


@pytest.mark.parametrize("stream", [False, True])
def test_sglang_sdk_chat_integration(stream):
    sglang_model = os.getenv("SGLANG_INTEGRATION_MODEL")
    sglang_api_base = os.getenv("SGLANG_INTEGRATION_API_BASE")
    if not sglang_model or not sglang_api_base:
        pytest.skip(
            "Set SGLANG_INTEGRATION_MODEL and SGLANG_INTEGRATION_API_BASE to run SGLang integration tests."
        )
    if not _is_sglang_ready(sglang_api_base):
        pytest.skip(f"SGLang server not ready at {sglang_api_base}")

    response = litellm.completion(
        model=f"sglang/{sglang_model}",
        api_base=sglang_api_base,
        api_key=os.getenv("SGLANG_INTEGRATION_API_KEY"),
        messages=[{"role": "user", "content": "Say hello in one word."}],
        stream=stream,
        max_tokens=16,
    )

    if stream:
        chunks = list(response)
        assert len(chunks) > 0
        assert chunks[-1].choices[0].finish_reason is not None
    else:
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason is not None
        assert response.usage is not None


def test_sglang_proxy_chat_integration():
    proxy_base = os.getenv("SGLANG_PROXY_BASE")
    proxy_model = os.getenv("SGLANG_PROXY_MODEL")
    proxy_key = os.getenv("SGLANG_PROXY_API_KEY", "sk-1234")
    if not proxy_base or not proxy_model:
        pytest.skip(
            "Set SGLANG_PROXY_BASE and SGLANG_PROXY_MODEL to run proxy integration tests."
        )

    payload = {
        "model": proxy_model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
    }
    response = httpx.post(
        f"{proxy_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {proxy_key}"},
        json=payload,
        timeout=20.0,
    )
    response.raise_for_status()
    data = response.json()

    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] is not None
    assert data.get("usage") is not None
