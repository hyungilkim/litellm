# SGLang

LiteLLM supports SGLang via its OpenAI-compatible `/v1/chat/completions` API.

| Property | Details |
|-------|-------|
| Provider Route on LiteLLM | `sglang/` |
| Supported Endpoints (initial scope) | `/chat/completions` |
| Environment Variables | `SGLANG_API_BASE`, `SGLANG_API_KEY` |

`api_base` should point to the SGLang OpenAI-compatible server base. LiteLLM normalizes both `http://host:30000` and `http://host:30000/v1` to `/v1`.

## SDK Usage

```python
from litellm import completion

response = completion(
    model="sglang/Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://localhost:30000",
)

print(response.choices[0].message.content)
```

## Proxy Usage

```yaml
model_list:
  - model_name: my-sglang-model
    litellm_params:
      model: sglang/Qwen/Qwen2.5-7B-Instruct
      api_base: http://sglang:30000
      api_key: os.environ/SGLANG_API_KEY
```
