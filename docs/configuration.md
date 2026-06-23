# Configuration

## Local Settings

Use `.mantis.toml` for both the LLM endpoint and Kaggle credentials:

```toml
provider = "openai"
use_gpu = true
online-evolution = true

[openai]
api_key = "<your-openai-compatible-api-key>"
base_url = "<your-openai-compatible-base-url>"
model = "mimo-v2.5-pro"
max_tokens = 8192
effort = "medium"

[kaggle]
username = "<your-kaggle-username>"
key = "<your-official-kaggle-api-key>"

[sandbox]
enabled = false
```

ACP mode can be used instead of direct API keys:

```toml
provider = "acp"
use_gpu = true
online-evolution = true

[acp]
agent = "codex"
cwd = "."
session = "mantis"
command = "acpx"
timeout = 1800
approve_all = false
# model = "gpt-5.2[high]"

[kaggle]
username = "<your-kaggle-username>"
key = "<your-official-kaggle-api-key>"
```

In ACP mode, Mantis calls `acpx --format json` and keeps a persistent named acpx
session. No LLM `api_key` or `base_url` is needed; the selected ACP-compatible
agent handles its own authentication and tool execution. Install acpx globally
with `npm install -g acpx@latest`, or set `command = "npx acpx@latest"`.

For a gateway-token Kaggle environment:

```toml
[kaggle]
kgat_api_token = "<your-kgat-token>"
```

Do not put a `KGAT_...` token in `key`. `key` is only for the official Kaggle API
key from Kaggle settings or `~/.kaggle/kaggle.json`; `KGAT_...` belongs in
`kgat_api_token`.

Mantis exports `[kaggle]` values as `KAGGLE_USERNAME`, `KAGGLE_KEY`,
`KGAT_API_TOKEN`, and `KAGGLE_API_TOKEN` when it starts.

`use_gpu = true` makes Mantis inspect available NVIDIA GPUs with `nvidia-smi` and
include GPU names, memory, and driver version in the system prompt. Sandbox is
off by default; `[sandbox] enabled = false` just records that explicitly.

`online-evolution = true` enables structured online competition learning. Kaggle
runs should maintain `competitions/<slug>/evolution/` and
`.mantis/online_evolution/`; global skill patches require cross-competition
evidence before changing `.mantis/skills/mantis-kaggle-distilled/`.

Environment variables also work and take priority over `.mantis.toml`:

```bash
export MANTIS_PROVIDER=openai
export MANTIS_API_KEY=<your-openai-compatible-api-key>
export MANTIS_BASE_URL=<your-openai-compatible-base-url>
export MANTIS_MODEL=mimo-v2.5-pro
export MANTIS_USE_GPU=true
export MANTIS_ONLINE_EVOLUTION=true
export MANTIS_ACP_AGENT=codex
export MANTIS_ACP_SESSION=mantis
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
| `MANTIS_MODEL` | Model name (e.g. `mimo-v2.5-pro`) |
| `MANTIS_MAX_TOKENS` | Max output tokens |
| `MANTIS_EFFORT` | Reasoning effort (`low`, `medium`, `high`) |
| `MANTIS_PROVIDER` | `anthropic` or `openai` |
| `MANTIS_API_KEY` | API key for OpenAI-compatible endpoints |
| `MANTIS_BASE_URL` | Base URL for OpenAI-compatible endpoints |
| `MANTIS_BUDDY_MODEL` | Model for companion pet reactions |
| `MANTIS_BUDDY_SEED` | Override buddy seed for specific companion |
| `MANTIS_USE_GPU` | Enable GPU-aware system prompt (`true`/`false`) |
| `MANTIS_ONLINE_EVOLUTION` | Enable online evolution artifacts and promotion gates (`true`/`false`) |
| `MANTIS_ACP_AGENT` | ACP agent passed to acpx, for example `codex` or `claude` |
| `MANTIS_ACP_SESSION` | Persistent acpx session name |
| `MANTIS_ACP_COMMAND` | acpx executable, for example `acpx` or `npx acpx@latest` |
| `MANTIS_ACP_TIMEOUT` | acpx turn timeout in seconds |
| `KAGGLE_USERNAME` | Kaggle username |
| `KAGGLE_KEY` | Kaggle API key |
| `KGAT_API_TOKEN` | Optional Kaggle gateway token |
| `KAGGLE_API_TOKEN` | Optional Kaggle gateway token alias |

## CLI Flags

```bash
mantis \
  --provider openai \
  --base-url <your-openai-compatible-base-url> \
  --api-key <your-openai-compatible-api-key> \
  --model mimo-v2.5-pro \
  --max-tokens 8192 \
  --auto-approve \
  --coordinator \
  --resume titanic
```

## Full-Access Permission Mode

Use this when you want Mantis to run without asking for per-tool permission:

```bash
mantis --auto-approve
```

For full local shell access, also disable sandbox in `.mantis.toml` or
`~/.config/mantis/config.toml`:

```toml
[sandbox]
enabled = false
```

Equivalent interactive command:

```text
/sandbox mode disabled
```

`--auto-approve` removes Mantis permission prompts. Disabling sandbox makes Bash
commands run directly instead of through bubblewrap isolation.

## Config File Loading

Loaded in order (later overrides earlier):

1. `~/.config/mantis/config.toml`
2. `.mantis.toml` in the current working directory

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

When `provider = "openai"`, `MANTIS_API_KEY` / `MANTIS_BASE_URL` or `OPENAI_API_KEY` / `OPENAI_BASE_URL` are used. When `provider = "anthropic"`, `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` are used.

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
