# AutoDS Configuration

This document describes how to configure the integrated AutoDS agent system and
its offline self-evolution corpus.

## Repository Layout

```text
.
├── src/                         # AutoDS agent runtime
├── tests/                       # Agent tests
├── docs/                        # Runtime documentation
├── .autods/skills/              # Project-local AutoDS skills
├── offline/                     # Offline self-evolution corpus and distillation
│   ├── Audio/ CV/ GenAI/ ...
│   ├── distillation/
│   ├── tools/
│   └── README.md
├── pyproject.toml
└── README.md
```

The top level is the runnable agent. Offline notebook data, manifests,
distillation reports, and corpus-building scripts live under `offline/`.

## Python Setup

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra installs test tooling, the Kaggle CLI, and baseline
data-science packages: `numpy`, `pandas`, and `scikit-learn`.

Run the CLI:

```bash
autods
autods "summarize this repository"
autods --auto-approve
autods --coordinator
```

## Local Configuration

AutoDS uses one local config file for the LLM endpoint, Kaggle credentials,
permission mode, and runtime settings. The default project-local file is
`.autods.toml`; a global config can also live at `~/.config/autods/config.toml`.

Recommended `.autods.toml`:

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
session = "autods"
command = "acpx"
timeout = 1800
approve_all = false
# model = "gpt-5.2[high]"

[kaggle]
username = "<your-kaggle-username>"
key = "<your-official-kaggle-api-key>"
```

In ACP mode, AutoDS calls `acpx --format json` and keeps a persistent named acpx
session. No LLM `api_key` or `base_url` is required in `.autods.toml`; the
selected ACP-compatible agent handles its own authentication and tool execution.
Use `command = "npx acpx@latest"` if `acpx` is not installed globally.

For a gateway-token Kaggle environment, use this instead of `username`/`key` if
that is how your local gateway is configured:

```toml
[kaggle]
kgat_api_token = "<your-local-gateway-token>"
```

Do not put a `KGAT_...` token in `key`. `key` is only for the official Kaggle API
key from Kaggle settings or `~/.kaggle/kaggle.json`; `KGAT_...` belongs in
`kgat_api_token`. AutoDS also treats `key = "KGAT_..."` as a gateway token for
backward compatibility, but the explicit field is clearer.

When AutoDS starts, `[kaggle]` values are exported into the process environment
as `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KGAT_API_TOKEN`, and `KAGGLE_API_TOKEN`.
Child shell commands and the `/kaggle` workflow inherit those values.

For Kaggle credentials, `.autods.toml` is the source of truth. If `[kaggle]`
contains `username` and official API `key`, AutoDS overwrites any old inherited
Kaggle gateway variables before the model runs. AutoDS skills should only use the
preconfigured environment; they should not run auth setup, write credential
files, read `~/.kaggle/access_token`, or export raw token values during a
competition workflow.

`use_gpu = true` asks AutoDS to inspect available NVIDIA GPUs with `nvidia-smi`
at startup and include the detected GPU names, memory, and driver version in the
system prompt. Omit it or set `use_gpu = false` for CPU-only work.

`online-evolution = true` enables structured online competition learning. When
enabled, Kaggle runs should maintain `competitions/<slug>/evolution/` and
`.autods/online_evolution/`. Global skill updates are evidence-gated and require
cross-competition support before patching `.autods/skills/autods-kaggle-distilled/`.

Sandbox is off by default. Keeping `[sandbox] enabled = false` makes that
explicit in local config.

Shell environment variables also work and take priority over `.autods.toml`:

```bash
export AUTODS_PROVIDER=openai
export AUTODS_API_KEY=<your-openai-compatible-api-key>
export AUTODS_BASE_URL=<your-openai-compatible-base-url>
export AUTODS_MODEL=mimo-v2.5-pro
export AUTODS_ONLINE_EVOLUTION=true
export AUTODS_ACP_AGENT=codex
export AUTODS_ACP_SESSION=autods
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-api-key>
```

Optional environment variables:

```bash
export AUTODS_MAX_TOKENS=8192
export AUTODS_EFFORT=medium
export AUTODS_BUDDY_MODEL=<optional-companion-model>
export AUTODS_MEMORY_DIR=$HOME/.config/autods/memory
export AUTODS_USE_GPU=true
export AUTODS_ONLINE_EVOLUTION=true
export AUTODS_ACP_COMMAND=acpx
export AUTODS_ACP_TIMEOUT=1800
```

Install Kaggle dependencies before using `/kaggle`:

```bash
pip install -e ".[dev]"
```

Kaggle's default credential file is still supported as a fallback:

```bash
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'JSON'
{"username":"<your-kaggle-username>","key":"<your-kaggle-api-key>"}
JSON
chmod 600 ~/.kaggle/kaggle.json
```

Do not commit `.env`, `.env.local`, `.autods.toml`, Kaggle credential files, or
raw API tokens.

## Full-Access Permission Mode

For exploration and experiments, you can start AutoDS in a mode that does not ask
for tool permission confirmations:

```bash
autods --auto-approve
```

For one-shot execution:

```bash
autods --auto-approve "summarize this repository"
```

For Kaggle work, start the terminal this way and then run `/kaggle` inside it:

```bash
autods --auto-approve
```

```text
/kaggle <competition-url-or-slug>
```

`--auto-approve` automatically approves AutoDS tool calls. It does not by itself
change sandbox behavior. If you want full local access for Bash commands as well,
keep sandbox disabled in `.autods.toml` or `~/.config/autods/config.toml`:

```toml
[sandbox]
enabled = false
```

You can also switch it from inside the terminal:

```text
/sandbox mode disabled
```

With `autods --auto-approve` plus `[sandbox] enabled = false`, AutoDS will not ask
for user permission before tool calls and Bash commands will run directly in the
local environment. Use this only in a workspace where you are comfortable letting
the agent read, write, install, submit, and run commands without per-action
confirmation.

## Skills

Project-local skills are stored in:

```text
.autods/skills/
```

The main project skill is:

```text
.autods/skills/autods-kaggle-distilled/SKILL.md
```

It points AutoDS to the V4 offline distillation under:

```text
offline/distillation/v4/
```

When working on a Kaggle competition, run:

```text
/kaggle <competition-url-or-slug>
```

The skill should prefer V4 deep-read distillation, then use V3/V2 as supporting
evidence.

## Offline Distillation Configuration

Run offline tools from the `offline/` directory:

```bash
cd offline
python3 tools/prepare_distillation_corpus.py
python3 tools/distill_v2_evolution_traces.py
python3 tools/build_v4_evidence_packs.py \
  --root . \
  --manifest distillation/coverage_manifest.csv \
  --out-dir distillation/v4
```

Important outputs:

```text
offline/distillation/reports/first_distillation_v1.md
offline/distillation/reports/offline_distillation_v2.md
offline/distillation/v2/
offline/distillation/v3/
offline/distillation/v4/deep_distillation_v4.md
offline/distillation/v4/evidence_packs_md/
```

The V4 corpus currently covers 45 selected competitions and 225 rank-bucket
files across 9 domains.

## Local Runtime Files

Common local runtime outputs:

```text
AUTODS.md
data/
submissions/
.env
.env.local
.autods.toml
```

These should stay local and are ignored by git.

## Verification

Basic checks:

```bash
python3 -m py_compile src/core/config.py src/core/engine.py src/features/skills_bundled.py
pytest -q
```

Offline symlink check:

```bash
python3 - <<'PY'
from pathlib import Path
broken = [p for p in Path("offline/distillation").rglob("*") if p.is_symlink() and not p.exists()]
print("broken symlinks:", len(broken))
PY
```
