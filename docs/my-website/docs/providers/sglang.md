# SGLang

https://docs.sglang.ai/basic_usage/openai_api.html

:::tip

SGLang exposes an OpenAI-compatible server. In LiteLLM, use the `sglang/` prefix to route requests to an SGLang deployment.

:::

## Required Variables

```python
import os

os.environ["SGLANG_API_BASE"] = "http://127.0.0.1:30000/v1"
# Optional. SGLang examples commonly use a dummy key like "EMPTY" / "None".
os.environ["SGLANG_API_KEY"] = "EMPTY"
```

## Usage - LiteLLM Python SDK

### Chat Completions

```python
from litellm import completion
import os

os.environ["SGLANG_API_BASE"] = "http://127.0.0.1:30000/v1"
os.environ["SGLANG_API_KEY"] = "EMPTY"

response = completion(
    model="sglang/Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "How many r's are in strawberry?"}],
    max_tokens=128,
)

print(response)
```

### Embeddings

```python
from litellm import embedding
import os

os.environ["SGLANG_API_BASE"] = "http://127.0.0.1:30000/v1"

response = embedding(
    model="sglang/Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    input=["LiteLLM routes requests to SGLang"],
)

print(response)
```

## Usage with LiteLLM Proxy

```yaml
model_list:
  - model_name: qwen3-4b
    litellm_params:
      model: sglang/Qwen/Qwen3-4B
      api_base: http://127.0.0.1:30000/v1
      api_key: EMPTY
```

Start the proxy:

```bash
litellm --config config.yaml
```

Then call it through the proxy:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "hello from litellm proxy"}]
  }'
```

## Notes

- Default LiteLLM route: `sglang/<model>`
- Default local SGLang server URL used by this integration: `http://127.0.0.1:30000/v1`
- `SGLANG_API_KEY` is optional for local deployments; LiteLLM will work with a dummy key if your server does not enforce auth.
