# Configuration

## Local Settings

Use `.autods.toml` for both the LLM endpoint and Kaggle credentials:

```toml
provider = "openai"
use_gpu = true

[openai]
api_key = "<your-openai-compatible-api-key>"
base_url = "<your-openai-compatible-base-url>"
model = "mimo-v2.5-pro"
max_tokens = 8192
effort = "medium"

[kaggle]
username = "<your-kaggle-username>"
key = "<your-kaggle-api-key>"

[sandbox]
enabled = false
```

For a gateway-token Kaggle environment:

```toml
[kaggle]
kgat_api_token = "<your-kgat-token>"
```

AutoDS exports `[kaggle]` values as `KAGGLE_USERNAME`, `KAGGLE_KEY`,
`KGAT_API_TOKEN`, and `KAGGLE_API_TOKEN` when it starts.

`use_gpu = true` makes AutoDS inspect available NVIDIA GPUs with `nvidia-smi` and
include GPU names, memory, and driver version in the system prompt. Sandbox is
off by default; `[sandbox] enabled = false` just records that explicitly.

Environment variables also work and take priority over `.autods.toml`:

```bash
export AUTODS_PROVIDER=openai
export AUTODS_API_KEY=<your-openai-compatible-api-key>
export AUTODS_BASE_URL=<your-openai-compatible-base-url>
export AUTODS_MODEL=mimo-v2.5-pro
export AUTODS_USE_GPU=true
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-api-key>
```

OpenAI-compatible aliases are also accepted:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://your-openai-gateway.example.com
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AUTODS_MODEL` | Model name (e.g. `mimo-v2.5-pro`) |
| `AUTODS_MAX_TOKENS` | Max output tokens |
| `AUTODS_EFFORT` | Reasoning effort (`low`, `medium`, `high`) |
| `AUTODS_PROVIDER` | `anthropic` or `openai` |
| `AUTODS_API_KEY` | API key for OpenAI-compatible endpoints |
| `AUTODS_BASE_URL` | Base URL for OpenAI-compatible endpoints |
| `AUTODS_BUDDY_MODEL` | Model for companion pet reactions |
| `AUTODS_BUDDY_SEED` | Override buddy seed for specific companion |
| `AUTODS_USE_GPU` | Enable GPU-aware system prompt (`true`/`false`) |
| `KAGGLE_USERNAME` | Kaggle username |
| `KAGGLE_KEY` | Kaggle API key |
| `KGAT_API_TOKEN` | Optional Kaggle gateway token |
| `KAGGLE_API_TOKEN` | Optional Kaggle gateway token alias |

## CLI Flags

```bash
autods \
  --provider openai \
  --base-url <your-openai-compatible-base-url> \
  --api-key <your-openai-compatible-api-key> \
  --model mimo-v2.5-pro \
  --max-tokens 8192 \
  --auto-approve \
  --coordinator \
  --resume 1
```

## Full-Access Permission Mode

Use this when you want AutoDS to run without asking for per-tool permission:

```bash
autods --auto-approve
```

For full local shell access, also disable sandbox in `.autods.toml` or
`~/.config/autods/config.toml`:

```toml
[sandbox]
enabled = false
```

Equivalent interactive command:

```text
/sandbox mode disabled
```

`--auto-approve` removes AutoDS permission prompts. Disabling sandbox makes Bash
commands run directly instead of through bubblewrap isolation.

## Config File Loading

Loaded in order (later overrides earlier):

1. `~/.config/autods/config.toml`
2. `.autods.toml` in the current working directory

Point to a specific file with `--config`.

### Alternative OpenAI Example

```toml
provider = "openai"

[openai]
api_key = "sk-..."
base_url = "https://your-openai-gateway.example.com/v1"
model = "gpt-4.1-mini"
max_tokens = 8192
effort = "medium"
buddy_model = "gpt-4.1-mini"
```

### OpenRouter (low-cost testing)

```toml
provider = "openai"

[openai]
api_key = "sk-or-..."
base_url = "https://openrouter.ai/api/v1"
model = "qwen/qwen3.6-plus-preview:free"
```

When `provider = "openai"`, `AUTODS_API_KEY` / `AUTODS_BASE_URL` or `OPENAI_API_KEY` / `OPENAI_BASE_URL` are used. When `provider = "anthropic"`, `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` are used.

Equivalent file-based setup:

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'JSON'
{"username":"<your-kaggle-username>","key":"<your-kaggle-api-key>"}
JSON
chmod 600 ~/.kaggle/kaggle.json
```

If your environment uses a gateway token, expose it as:

```bash
export KGAT_API_TOKEN=<your-kgat-token>
export KAGGLE_API_TOKEN=<your-kgat-token>
```

The official Kaggle CLI still requires `KAGGLE_USERNAME` and `KAGGLE_KEY` unless
a local wrapper or gateway in your environment explicitly supports
`KGAT_API_TOKEN`.
