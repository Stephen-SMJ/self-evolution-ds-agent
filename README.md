# AutoDS

AutoDS is an experimental self-evolving data science agent for Kaggle-style competitions.

This repository contains two parts:

- **Agent system**: the runnable AutoDS terminal agent, tools, slash commands, project skills, and tests.
- **Offline evolution corpus**: collected Kaggle notebooks and V1-V4 distillation artifacts under `offline/`.

The project goal is to turn competition experience into reusable agent behavior. AutoDS should be able to receive a Kaggle competition link, understand the task and metric, download data, build baselines, run experiments, submit predictions, inspect scores, reflect on failures, and improve its strategy over time.

## Why This Exists

Most data-science agents fail because they treat every competition as a generic modeling task. Strong Kaggle solutions usually depend on details that are easy to miss:

- the exact metric and submission format
- leakage-safe validation
- domain-specific feature construction
- OOF diagnostics and public/private leaderboard gaps
- task-specific postprocessing
- disciplined ensembling
- knowing when a public notebook is low-quality or not transferable

AutoDS is built around those details. The offline corpus is used as prior knowledge, while the agent runtime is used to execute and iterate on real competitions.

## Current Status

This is an active research codebase, not a polished product.

Implemented:

- interactive terminal agent runtime
- OpenAI-compatible model configuration
- file, shell, search, edit, and planning tools
- project-local skills
- `/kaggle` competition workflow skill
- offline notebook corpus
- V1-V4 distillation reports
- V4 deep-read competition playbooks

Still evolving:

- stronger online competition loop
- automatic experiment tracking beyond `AUTODS.md`
- robust notebook execution and replay
- V5 validation by actually running selected notebooks/experiments
- domain-specific skill expansion after more offline distillation

## Repository Layout

```text
.
├── src/                         # AutoDS agent runtime
│   ├── core/                    # config, LLM client, engine, sessions
│   ├── tools/                   # Read/Edit/Write/Grep/Glob/Bash/etc.
│   ├── features/                # skills, memory, planner, coordinator, sandbox
│   ├── commands/                # slash command handlers
│   ├── tui/                     # terminal UI
│   └── buddy/                   # optional terminal companion feature
├── .autods/skills/              # project-local skills used by AutoDS
│   └── autods-kaggle-distilled/ # Kaggle/self-evolution skill
├── agentic-kaggle-skill/        # reference skill material
├── docs/                        # agent runtime docs
├── tests/                       # runtime tests
├── offline/                     # offline self-evolution corpus
│   ├── Audio/
│   ├── CV/
│   ├── GenAI/
│   ├── Medical/
│   ├── NLP/
│   ├── RL/
│   ├── RecSys/
│   ├── Tabular/
│   ├── Time-Series/
│   ├── distillation/
│   ├── tools/
│   └── README.md
├── CONFIGURATION.md             # full setup/config guide
├── pyproject.toml
└── README.md
```

The top level is the runnable agent. Everything related to notebook collection, ranking buckets, corpus manifests, and offline distillation lives in `offline/`.

## Offline Distillation

The offline stage studies public Kaggle notebooks across domains and rank buckets. The aim is not to copy notebooks. The aim is to extract transferable upgrade mechanisms from weaker solutions to stronger ones.

Current V4 coverage:

- 9 domains
- 45 selected competitions
- 225 rank-bucket files
- 5 buckets per selected competition: `70pct`, `40pct`, `20pct`, `10pct`, `1st`

Domain coverage:

| Domain | Selected Competitions |
| --- | ---: |
| Audio | 5 |
| CV | 5 |
| GenAI | 5 |
| Medical | 5 |
| NLP | 5 |
| RL | 5 |
| RecSys | 5 |
| Tabular | 5 |
| Time-Series | 5 |

Example: `titanic` is included in the Tabular offline corpus with all five rank
buckets (`70pct`, `40pct`, `20pct`, `10pct`, `1st`) and V4 evidence packs under
`offline/distillation/v4/evidence_packs_md/Tabular/titanic.md`.

Important files:

```text
offline/distillation/reports/first_distillation_v1.md
offline/distillation/reports/offline_distillation_v2.md
offline/distillation/v2/
offline/distillation/v3/
offline/distillation/v4/deep_distillation_v4.md
offline/distillation/v4/deep_traces_batch1_tabular_timeseries_nlp.md
offline/distillation/v4/deep_traces_batch2_cv_medical_audio.md
offline/distillation/v4/deep_traces_batch3_recsys_genai_rl.md
offline/distillation/v4/evidence_packs/
offline/distillation/v4/evidence_packs_md/
```

### V1

V1 is the first manual overview. It records early observations and rough lessons from the initial corpus.

Use it for historical context only.

### V2

V2 turns the corpus into structured manifests and rule-based traces.

It detects patterns such as:

- validation: KFold, GroupKFold, OOF, time split
- features: aggregation, lagging, encoding, text/image/audio transforms
- models: GBM, CNN, transformers, linear baselines
- training: augmentation, early stopping, schedulers
- postprocessing: thresholds, clipping, ranking, blending
- risk: random submissions, downloader-only notebooks, thin artifacts

V2 is useful as an index, but it is not enough for strategy.

### V3

V3 adds semantic comparison of rank buckets. It is more useful than V2 for identifying upgrade paths, but the current committed V3 is mainly a semantic report, not a fully automated training pipeline.

### V4

V4 is the main planning-quality distillation layer.

It is based on evidence packs extracted from notebook/code cells and then manually analyzed competition by competition. It explicitly marks low-quality evidence, including:

- downloader-only notebooks
- random/sample submissions
- leaderboard-only analysis
- very short artifacts
- thin ensemble files without training logic
- high-rank buckets that are not actually good modeling examples

The most important V4 lesson:

> Strong solutions are usually not just stronger models. They come from metric alignment, validation design, representation construction, domain features, OOF diagnostics, postprocessing, and controlled ensembling.

## Agent Runtime

AutoDS runs as a terminal agent. It can read and edit files, run shell commands with permission checks, search the repository, ask the user questions, use project skills, and maintain session context.

The Kaggle workflow is implemented as a project-local skill:

```text
.autods/skills/autods-kaggle-distilled/SKILL.md
```

When `/kaggle <competition>` is used, the skill points AutoDS to the offline V4 playbooks and asks the agent to:

1. identify competition slug, task type, metric, files, and submission format
2. create or update `AUTODS.md`
3. build the simplest valid baseline
4. choose leakage-aware validation
5. record CV/LB gaps and experiment decisions
6. apply domain playbooks from V4
7. avoid copying notebooks blindly
8. write reusable lessons back into the project workflow

## Installation

Use Python 3.11 or newer.

```bash
git clone git@github.com:Stephen-SMJ/self-evolution-ds-agent.git
cd self-evolution-ds-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra includes the Kaggle CLI plus baseline data-science packages such
as `numpy`, `pandas`, and `scikit-learn`.

## Local Configuration

Keep the LLM API and Kaggle credentials together in a local `.autods.toml`:

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
key = "<your-official-kaggle-api-key>"

[sandbox]
enabled = false
```

For a gateway-token Kaggle environment:

```toml
[kaggle]
kgat_api_token = "<your-local-gateway-token>"
```

Do not put a `KGAT_...` token in `key`. `key` is for the official Kaggle API key
from Kaggle settings or `~/.kaggle/kaggle.json`; `KGAT_...` belongs in
`kgat_api_token`.

When AutoDS starts, it exports `[kaggle]` values into the process environment as
`KAGGLE_USERNAME`, `KAGGLE_KEY`, `KGAT_API_TOKEN`, and `KAGGLE_API_TOKEN`, so the
`/kaggle` workflow and child shell commands can use them. Existing shell
environment variables take priority over `.autods.toml`.

Environment variables also work:

```bash
export AUTODS_PROVIDER=openai
export AUTODS_API_KEY=<your-openai-compatible-api-key>
export AUTODS_BASE_URL=<your-openai-compatible-base-url>
export AUTODS_MODEL=mimo-v2.5-pro
export AUTODS_USE_GPU=true
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-api-key>
```

Install the Kaggle CLI dependency before using `/kaggle`:

```bash
pip install -e ".[dev]"
```

Do not commit `.autods.toml`, `.env`, Kaggle credential files, API keys, or
tokens.

`use_gpu = true` tells AutoDS to inspect available NVIDIA GPUs at startup and add
the detected GPU names, memory, and driver version to the system prompt. Keep it
`false` or omit it on CPU-only machines.

Sandbox is disabled by default. Keeping `[sandbox] enabled = false` in
`.autods.toml` makes that explicit and lets Bash commands run directly in the
local workspace.

Full configuration guide:

```text
CONFIGURATION.md
```

## Full-Access Mode

For exploratory experiments where you do not want AutoDS to ask for every tool permission, start it with automatic approval:

```bash
autods --auto-approve
```

For a Kaggle workflow:

```bash
autods --auto-approve
```

Then run inside the AutoDS terminal:

```text
/kaggle <competition-url-or-slug>
```

`--auto-approve` means AutoDS will approve tool calls without prompting you. To also run Bash commands without sandbox isolation, keep sandbox disabled:

```toml
[sandbox]
enabled = false
```

The same setting can be changed in the interactive terminal:

```text
/sandbox mode disabled
```

This is the closest local equivalent of full-access mode: no permission prompts from AutoDS and no sandbox wrapping for shell commands.

## Running AutoDS

Interactive mode:

```bash
autods
```

One-shot mode:

```bash
autods "summarize the offline V4 distillation"
```

Kaggle workflow:

```text
/kaggle https://www.kaggle.com/competitions/titanic
```

Competition sub-pages work too. AutoDS routes `/kaggle` commands through the
Kaggle skill and normalizes URLs like this to the competition slug `titanic`:

```text
/kaggle https://www.kaggle.com/competitions/titanic/leaderboard
```

The same normalization handles common Kaggle competition pages such as `/data`,
`/code`, `/rules`, `/discussion`, and `/submissions`.

After a baseline is submitted and scored, AutoDS should continue the
self-evolution loop by default: inspect the CV/LB gap, read any matching offline
distillation evidence, run the next high-ROI experiment, and update
`competitions/<slug>/AUTODS.md`. It should not stop with "Want me to continue?"
unless it is blocked, at a submission/rules risk point, or the user explicitly
asks it to pause.

Useful slash commands:

```text
/skills
/kaggle <competition-url-or-slug>
/review
/test
/commit
/compact
/resume
```

## Regenerating Offline Artifacts

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

V3 API-driven semantic distillation requires a valid OpenAI-style endpoint:

```bash
cd offline
python3 tools/distill_v3_llm_semantic.py --root .
```

## Testing

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

## Development Notes

- Keep the root directory focused on the runnable agent.
- Keep raw notebooks, corpus manifests, and distillation outputs under `offline/`.
- Do not commit API keys, Kaggle credentials, generated submissions, local data, or `.env` files.
- Treat public notebooks as evidence, not truth.
- Prefer V4 distilled playbooks over raw notebook copying.
- When evidence is weak, mark it explicitly instead of forcing it into a recipe.

## Roadmap

Near-term:

- improve `/kaggle` experiment logging
- add reusable domain skill files generated from V4
- add notebook quality scoring beyond keyword/rank buckets
- add execution-based validation for selected notebooks
- create a V5 pass that verifies distilled upgrades by running controlled experiments

Longer-term:

- online self-evolution loop for active Kaggle competitions
- automatic competition-specific skill writing
- domain memory for tabular, CV, NLP, audio, medical, recsys, GenAI, and RL/game tasks
- stronger public/private leaderboard gap analysis
