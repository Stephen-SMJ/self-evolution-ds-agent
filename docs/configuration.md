# Configuration

## API Settings

### AutoDS gateway (default)

```bash
export AUTODS_PROVIDER=openai
export AUTODS_API_KEY=<your-openai-compatible-api-key>
export AUTODS_BASE_URL=<your-openai-compatible-base-url>
export AUTODS_MODEL=mimo-v2.5-pro
```

### OpenAI-compatible override

```bash
export AUTODS_PROVIDER=openai
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

## TOML Config Files

Loaded in order (later overrides earlier):

1. `~/.config/autods/config.toml`
2. `.autods.toml` in the current working directory

Point to a specific file with `--config`.

### AutoDS gateway example

```toml
provider = "openai"

[openai]
api_key = "<your-openai-compatible-api-key>"
base_url = "<your-openai-compatible-base-url>"
model = "mimo-v2.5-pro"
```

### AutoDS gateway + Kaggle example

```toml
provider = "openai"

[openai]
api_key = "<your-openai-compatible-api-key>"
base_url = "<your-openai-compatible-base-url>"
model = "mimo-v2.5-pro"
max_tokens = 8192
effort = "medium"

[kaggle]
username = "<your-kaggle-username>"
key = "<your-kaggle-api-key>"
```

### OpenAI example

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

## Kaggle Credentials

AutoDS's `/kaggle` workflow uses the official Kaggle CLI when it needs to list
files, download competition data, submit predictions, or manage notebooks.

Preferred local setup in `.autods.toml`:

```toml
[kaggle]
username = "<your-kaggle-username>"
key = "<your-kaggle-api-key>"
```

AutoDS exports these values as `KAGGLE_USERNAME` and `KAGGLE_KEY` when it starts.
Existing shell environment variables take priority over `.autods.toml`.

Gateway-token setup:

```toml
[kaggle]
kgat_api_token = "<your-kgat-token>"
```

`kgat_api_token` is exported as both `KGAT_API_TOKEN` and `KAGGLE_API_TOKEN`.

Shell environment variables also work:

```bash
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-api-key>
```

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
