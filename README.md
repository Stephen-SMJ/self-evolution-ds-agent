<div align="center">

# AutoDS

**Self-evolving agent harness for data science**

**Agentic** &nbsp;·&nbsp; **Data Science Focused** &nbsp;·&nbsp; **Built to Extend**
<br>

This repository contains the AutoDS agent system at the top level and the
offline self-evolution corpus under `offline/`.

</div>

---

### **NEW: Buddy — AI Companion with Custom Sprites**

> Your coding companion lives in the terminal. Type `/buddy` to hatch it. Supports custom ASCII species — bring your own Pikachu!

![Custom Pikachu buddy companion](assets/buddy-pikachu.jpg)

[Full Buddy docs &rarr;](docs/buddy.md)

---

## Features

### Core

- **Interactive REPL** with streaming output, command history, slash command autocomplete
- **Agentic tool loop** — AutoDS calls tools autonomously until the task is complete
- **9 built-in tools**: `Read`, `Edit`, `Write`, `Glob`, `Grep`, `Bash`, `AskUser`, `EnterPlanMode`, `ExitPlanMode`
- **Plan mode** — parallel subagents explore codebase before you implement, with permission isolation
- **Permission system** — mode-aware (default/plan), reads auto-approved, writes/bash ask for confirmation
- **Session persistence** — auto-save conversations, `/resume` to continue later
- **Context compression** — auto-compact when approaching token limits
- **OpenAI-compatible runtime** — configure with `AUTODS_API_KEY`, `AUTODS_BASE_URL`, and `AUTODS_MODEL`
- **Kaggle competition workflow** — `/kaggle` guides competition intake, data download, baselines, submissions, notebook workflows, and self-evolution logs
- **Offline self-evolution prior** — V1-V4 notebook distillation reports and evidence packs live in `offline/`

### Advanced AutoDS capabilities

| Feature | Description | Docs |
|---------|-------------|------|
| **Coordinator Mode** | Background workers for parallel research and implementation | [docs &rarr;](docs/coordinator.md) |
| **Buddy** | Tamagotchi AI pet with personality, stats, mood, and speech bubbles | [docs &rarr;](docs/buddy.md) |
| **KAIROS Memory** | Cross-session memory with auto-consolidation | [docs &rarr;](docs/memory.md) |
| **Skills** | One-command workflows: `/review`, `/commit`, `/test`, `/simplify` | [docs &rarr;](docs/skills.md) |
| **Sandbox** | Bubblewrap isolation for bash commands | [docs &rarr;](docs/sandbox.md) |

See [docs/examples/citorigin](docs/examples/citorigin/README.md) for a
project-specific custom skill example.

---

## Quick Start

### Requirements

- Python 3.11+
- An OpenAI-compatible API endpoint

### Install

```bash
cd self-evolution-ds-agent
pip install -e ".[dev]"
```

### API Settings

```bash
export AUTODS_PROVIDER=openai
export AUTODS_API_KEY=<your-openai-compatible-api-key>
export AUTODS_BASE_URL=<your-openai-compatible-base-url>
export AUTODS_MODEL=mimo-v2.5-pro
```

### Run

```bash
autods                              # Interactive REPL
autods "what tests exist?"          # One-shot prompt
autods -p "summarize this codebase" # Print and exit
autods --auto-approve               # Skip permission prompts
autods --resume 1                   # Resume previous session
autods --coordinator                # Coordinator mode
```

### First Session Demo

```
autods

> list all python files in this project
↳ Glob(**/*.py) ✓
Found 12 Python files...

> read engine.py and explain the tool loop
↳ Read(src/core/engine.py) ✓
The submit() method implements an agentic loop...

> /buddy
Hatching your companion...
✨ SHINY LEGENDARY DUCK
Glitch Quack hatched! ★★★★★

> /buddy mood
Glitch Quack's mood:
  Happy      ████████████████░░░░  65 (high)
  Bored      ██████████░░░░░░░░░░  50 (neutral)

> /review
Running skill: /review…
↳ Bash(git diff) … ✓ done
## Code Review: no issues found ✓
```

[Full configuration docs &rarr;](CONFIGURATION.md)

---

## Tools

| Tool | Description | Permission |
|------|-------------|------------|
| `Read` | Read file contents | auto-approved |
| `Glob` | Find files by pattern | auto-approved |
| `Grep` | Search file contents | auto-approved |
| `Edit` | Edit file (string replacement) | requires confirmation |
| `Write` | Write/create file | requires confirmation |
| `Bash` | Run shell command | requires confirmation |
| `AskUser` | Ask user a question | auto-approved |
| `EnterPlanMode` | Enter plan mode | auto-approved |
| `ExitPlanMode` | Exit plan mode | auto-approved |

Coordinator mode adds: `Agent` (spawn worker), `SendMessage` (continue worker), `TaskStop` (stop worker). Plan mode also uses `Agent` to launch parallel read-only explore/plan subagents. See [coordinator docs](docs/coordinator.md).

---

## Data Paths

| Data | Path |
|------|------|
| Installation (source code) | `~/.autods/` |
| Sessions | `~/.config/autods/sessions/` |
| Memory (KAIROS) | `~/.config/autods/memory/` |
| Plans | `~/.config/autods/plans/` |
| REPL history | `~/.config/autods/history` |
| Companion data | `~/.config/autods/companion.json` |
| User skills | `~/.autods/skills/` |
| Project skills | `{cwd}/.autods/skills/` |
| Project config | `.autods.toml` |

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/compact` | Compress conversation context |
| `/resume` | Resume a past session |
| `/history` | List saved sessions |
| `/clear` | Clear conversation, start new session |
| `/skills` | List all available skills |
| `/kaggle` | Run the general Kaggle competition workflow |
| `/buddy` | Companion pet — hatch, pet, stats, mood |
| `/buddy help` | Show all buddy commands and gameplay guide |
| `/review` | Code review (skill) |
| `/commit` | Git commit (skill) |
| `/test` | Run tests (skill) |
| `/simplify` | Review and fix code (skill) |

Type `/` to see autocomplete suggestions.

---

## Project Structure

```
src/
├── core/                  # Pure harness — engine, LLM, config
│   ├── engine.py          # Streaming API loop + tool execution
│   ├── llm.py             # LLM client (Anthropic + OpenAI)
│   ├── config.py          # Configuration (CLI, env, TOML)
│   ├── context.py         # System prompt builder
│   ├── tool.py            # Base Tool protocol + ToolResult
│   ├── permissions.py     # Permission checker
│   └── session.py         # Session persistence
│
├── tools/                 # Tool implementations (one per file)
│   ├── bash.py            # Shell command execution
│   ├── file_read.py       # Read files
│   ├── file_edit.py       # Edit files (string replacement)
│   ├── file_write.py      # Write/create files
│   ├── glob_tool.py       # Find files by pattern
│   ├── grep_tool.py       # Search file contents
│   ├── ask_user.py        # Ask user questions
│   ├── plan_tools.py      # EnterPlanMode / ExitPlanMode
│   └── agent.py           # Coordinator agent tools
│
├── features/              # Pluggable capabilities
│   ├── compact.py         # Context compression
│   ├── coordinator.py     # Coordinator mode
│   ├── worker_manager.py  # Background worker lifecycle
│   ├── cost_tracker.py    # Token usage tracking
│   ├── memory.py          # KAIROS memory system
│   ├── plan.py            # Plan mode logic
│   ├── skills.py          # Skill loader and registry
│   ├── skills_bundled.py  # Built-in skills (review, commit, test, simplify)
│   └── sandbox/           # Bubblewrap sandbox subsystem
│
├── tui/                   # Terminal UI
│   ├── app.py             # CLI entry point + REPL
│   ├── query.py           # Query submission + streaming display
│   ├── rendering.py       # Rich console rendering
│   ├── prompt.py          # Input prompt
│   ├── input_parser.py    # Input parsing
│   ├── shell.py           # Shell integration
│   └── keylistener.py     # Esc/Ctrl+C detection
│
├── commands/              # Slash command handlers
└── buddy/                 # AI companion pet system
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v -k "not integration"  # skip bwrap tests
```

---

## Documentation

| Topic | Link |
|-------|------|
| Configuration (API keys, TOML, CLI flags) | [docs/configuration.md](docs/configuration.md) |
| Buddy (AI companion pet) | [docs/buddy.md](docs/buddy.md) |
| Coordinator Mode (background workers) | [docs/coordinator.md](docs/coordinator.md) |
| KAIROS Memory System | [docs/memory.md](docs/memory.md) |
| Skills (custom workflows) | [docs/skills.md](docs/skills.md) |
| Sandbox (bash isolation) | [docs/sandbox.md](docs/sandbox.md) |
