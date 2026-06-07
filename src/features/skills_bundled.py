"""Bundled skills — built-in skills shipped with autods.

Modelled after AutoDS's ``src/skills/bundled/`` directory.
Each skill is registered via ``register_skill()`` during startup.
"""

from __future__ import annotations

from .skills import Skill, register_skill


# ---------------------------------------------------------------------------
# /simplify — Code review and cleanup
# ---------------------------------------------------------------------------

_SIMPLIFY_PROMPT = """\
# Simplify: Code Review and Cleanup

Review all changed files for reuse, quality, and efficiency. Fix any issues found.

## Phase 1: Identify Changes

Run `git diff` (or `git diff HEAD` if there are staged changes) to see what \
changed. If there are no git changes, review the most recently modified files \
that the user mentioned or that you edited earlier in this conversation.

## Phase 2: Review

Examine each changed file for:

### Code Reuse
- Duplicated logic that should be extracted into shared functions
- Existing utilities or helpers in the codebase that could replace new code
- Patterns that appear more than twice and should be abstracted

### Code Quality
- Unclear variable or function names
- Missing or incorrect type annotations
- Overly complex logic that could be simplified
- Dead code or unused imports
- Inconsistent style with the rest of the codebase

### Efficiency
- Unnecessary allocations or copies
- N+1 patterns or repeated lookups
- Missing early returns or short-circuit evaluations
- Opportunities to use more efficient data structures

## Phase 3: Fix Issues

For each issue found, fix it directly in the code. Do not just list issues — \
apply the fixes. After fixing, run any relevant tests or linters to verify \
the changes don't break anything.

$ARGUMENTS\
"""


def _simplify_prompt(args: str) -> str:
    text = _SIMPLIFY_PROMPT
    if args:
        text = text.replace("$ARGUMENTS",
                            f"\n## Additional Focus\n\n{args}")
    else:
        text = text.replace("$ARGUMENTS", "")
    return text


# ---------------------------------------------------------------------------
# /review — Code review without auto-fix
# ---------------------------------------------------------------------------

_REVIEW_PROMPT = """\
# Code Review

Review the recent code changes and provide detailed feedback. Do NOT make \
changes — only analyze and report.

## Steps

1. Run `git diff` (or `git diff HEAD` for staged changes) to see what changed.
2. For each changed file, review for:
   - Correctness: logic errors, edge cases, off-by-one errors
   - Security: injection vulnerabilities, unsafe operations, exposed secrets
   - Performance: inefficient patterns, unnecessary work
   - Readability: unclear naming, missing context, complex logic
   - Style: consistency with codebase conventions
3. Provide a structured report with findings grouped by severity:
   - **Critical** — bugs or security issues that must be fixed
   - **Warning** — issues that should be addressed
   - **Suggestion** — improvements that would be nice to have

$ARGUMENTS\
"""


def _review_prompt(args: str) -> str:
    text = _REVIEW_PROMPT
    if args:
        text = text.replace("$ARGUMENTS",
                            f"\n## Additional Focus\n\n{args}")
    else:
        text = text.replace("$ARGUMENTS", "")
    return text


# ---------------------------------------------------------------------------
# /commit — Generate commit message and commit
# ---------------------------------------------------------------------------

_COMMIT_PROMPT = """\
# Git Commit

Create a well-structured git commit for the current staged changes.

## Steps

1. Run `git status` to see what is staged and unstaged.
2. Run `git diff --cached` to see staged changes. If nothing is staged, run \
`git diff` to see unstaged changes and inform the user.
3. Analyze the changes and create a commit message following conventional \
commit style:
   - First line: concise summary (50 chars max), imperative mood
   - Blank line
   - Body: explain what and why (not how), wrap at 72 chars
4. Run `git commit -m "<message>"` with the generated message.

If the user provided instructions, incorporate them into the commit message.

$ARGUMENTS\
"""


def _commit_prompt(args: str) -> str:
    text = _COMMIT_PROMPT
    if args:
        text = text.replace("$ARGUMENTS",
                            f"\n## User Instructions\n\n{args}")
    else:
        text = text.replace("$ARGUMENTS", "")
    return text


# ---------------------------------------------------------------------------
# /test — Run and analyze tests
# ---------------------------------------------------------------------------

_TEST_PROMPT = """\
# Run Tests

Find and run the project's test suite, then analyze the results.

## Steps

1. Identify the test framework:
   - Look for `pytest.ini`, `pyproject.toml` [tool.pytest], `setup.cfg`
   - Look for `package.json` scripts (test, jest, vitest)
   - Look for `Makefile` test targets
2. Run the appropriate test command.
3. If tests fail:
   - Analyze each failure
   - Identify the root cause
   - Suggest or apply fixes if the failures are in recently changed code

$ARGUMENTS\
"""


def _test_prompt(args: str) -> str:
    text = _TEST_PROMPT
    if args:
        text = text.replace("$ARGUMENTS",
                            f"\n## Specific Instructions\n\n{args}")
    else:
        text = text.replace("$ARGUMENTS", "")
    return text


# ---------------------------------------------------------------------------
# /kaggle — General Kaggle competition workflow
# ---------------------------------------------------------------------------

_KAGGLE_PROMPT = """\
# Kaggle Competition Workflow

Use this skill to run the first version of AutoDS's general Kaggle competition workflow.
The user may provide a Kaggle URL, competition slug, local competition folder, notebook,
or a broad request such as "start this competition".

## Goal

Act as a self-evolving data science competition agent. Build a valid baseline,
iterate toward stronger scores, and record reusable lessons as skills or memory.

## Offline-Distilled Prior

Before making non-trivial modeling decisions, check whether the project-local
distilled skill exists:

- `.autods/skills/autods-kaggle-distilled/SKILL.md`
- `.autods/skills/autods-kaggle-distilled/references/general-workflow.md`
- `.autods/skills/autods-kaggle-distilled/references/domain-playbooks.md`
- `.autods/skills/autods-kaggle-distilled/references/evolution-rubric.md`
- `.autods/skills/autods-kaggle-distilled/references/v2-distilled-recipes.md`

If present, use it as the default competition prior. In particular:

- use V2 `domain_recipes` and `competition_evolution_traces` as the first retrieval layer
- use `core_manifest.csv` / `core_files` as the main offline distilled reference
- use `coverage_manifest.csv` / `coverage_files` only for missing domain/rank coverage
- apply the domain playbook matching the competition type
- use the evolution rubric to move from valid baseline to stronger solution
- record which distilled lesson was used in `AUTODS.md`

## Phase 1: Competition Intake

1. Parse the competition slug from the user input when possible:
   - `https://www.kaggle.com/competitions/<slug>`
   - `<slug>`
2. Inspect or fetch competition information:
   - task type and domain: tabular regression/classification, CV, NLP, audio, time series,
     recommender, LLM, RL/game, optimization, or other
   - evaluation metric and whether higher or lower is better
   - submission format and required columns from `sample_submission.csv`
   - data files, train/test split, target column, ID columns, leakage risks
   - rules: internet, external data, notebook/code requirements, submission limits
3. Create or reuse an isolated workspace:
   - `competitions/<slug>/AUTODS.md`
   - `competitions/<slug>/data/`
   - `competitions/<slug>/src/`
   - `competitions/<slug>/notebooks/`
   - `competitions/<slug>/experiments/`
   - `competitions/<slug>/submissions/`
   - `competitions/<slug>/outputs/`
4. Create or update a compact `competitions/<slug>/AUTODS.md` with:
   - competition overview
   - metric and validation plan
   - data inventory
   - baseline plan
   - experiment log table
   - submission checklist

## Phase 2: Environment and Credentials

Prefer the official Kaggle CLI when available.

Useful commands:

```bash
kaggle competitions files -c <slug>
kaggle competitions download -c <slug> -p competitions/<slug>/data
kaggle competitions submissions -c <slug>
kaggle kernels list --competition <slug> --sort-by hotness
kaggle kernels pull <owner>/<kernel> -p competitions/<slug>/notebooks/<kernel> -m
```

Dependency policy:
- Prefer packages already installed in the current project virtualenv.
- If a dependency is missing, install only into the current project virtualenv:
  `.venv/bin/python -m pip install <package>` from the AutoDS project root.
- Do not use global `pip`, `pip3`, `sudo pip`, apt, conda, or
  `--break-system-packages` unless the user explicitly asks for system-level changes.
- If `.venv` is unavailable, ask before changing the Python environment.

Credential checks:

```bash
python - <<'PY'
import os
print("KAGGLE_USERNAME:", bool(os.getenv("KAGGLE_USERNAME")))
print("KAGGLE_KEY:", bool(os.getenv("KAGGLE_KEY")))
print("KAGGLE_CONFIG_DIR:", os.getenv("KAGGLE_CONFIG_DIR"))
print("KGAT_API_TOKEN:", bool(os.getenv("KGAT_API_TOKEN")))
PY
```

If credentials are missing, tell the user that Kaggle CLI normally needs
`KAGGLE_USERNAME` and `KAGGLE_KEY` or `~/.kaggle/kaggle.json`. If the environment
uses `KGAT_API_TOKEN`, use it only through the supported gateway or wrapper
available in the local project; do not invent a Kaggle username from it.

## Phase 3: Baseline First

Build the fastest valid baseline before optimizing:

1. Load train/test/sample submission.
2. Validate row counts, ID alignment, required columns, dtypes, missing values, and target.
3. Create a simple baseline:
   - tabular: constant/mean, LightGBM/XGBoost/CatBoost if available, or sklearn fallback
   - CV/audio/NLP: sample-submission-safe placeholder or lightweight feature baseline
   - time series: naive/grouped lag baseline
4. Write `competitions/<slug>/submissions/submission_baseline.csv`.
5. Locally validate format:

```python
import pandas as pd
sub = pd.read_csv("competitions/<slug>/submissions/submission_baseline.csv")
sample = pd.read_csv("competitions/<slug>/data/sample_submission.csv")
assert list(sub.columns) == list(sample.columns)
assert len(sub) == len(sample)
assert sub.iloc[:, 0].equals(sample.iloc[:, 0])
assert not sub.isna().any().any()
```

## Phase 4: Submission and Monitoring

Before submitting, ask for confirmation unless the user explicitly allowed automatic submissions.

Prediction submission:

```bash
kaggle competitions submit -c <slug> -f competitions/<slug>/submissions/submission_baseline.csv -m "AutoDS baseline"
kaggle competitions submissions -c <slug>
```

Notebook/kernel workflow:

```bash
kaggle kernels pull <owner>/<kernel> -p competitions/<slug>/notebooks/<kernel> -m
kaggle kernels push -p competitions/<slug>/notebooks/<kernel>
kaggle kernels status <username>/<kernel-name>
kaggle kernels output <username>/<kernel-name> -p competitions/<slug>/outputs/<kernel-name>
```

For code competitions or notebook-required submissions, preserve Kaggle metadata:
`competition_sources`, `dataset_sources`, `kernel_sources`, `model_sources`,
`enable_internet`, GPU settings, and private/public visibility.

When competition rules require notebook linkage, use `-k <kernel> -v <version>` on
submission commands if supported by the current Kaggle CLI version.

## Phase 5: Iteration and Self-Evolution

After each experiment, update the experiment log:

| run | change | local CV | public score | status | lesson | next |
|-----|--------|----------|--------------|--------|--------|------|

Use a disciplined loop:

1. Diagnose current bottleneck: validation mismatch, feature quality, model capacity,
   ensembling, post-processing, data leakage, runtime, memory, or submission format.
2. Run one high-leverage change at a time.
3. Compare local validation and public leaderboard.
4. Keep winning changes, revert or isolate losing changes.
5. Extract reusable lessons into `AUTODS.md`, a project skill, or memory.

Score handling:
- Submit early to establish a baseline.
- Do not over-trust very early public scores on competitions with delayed evaluation.
- Wait for completion/stabilization before making major strategic decisions.
- Track both local validation and public score to detect leaderboard overfitting.

## Troubleshooting Patterns

400 Bad Request:
- Check header, columns, row count, ID values, quote/CSV formatting, and file extension.
- Try zipping only the required CSV:

```python
import zipfile
with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write("submission.csv", "submission.csv")
```

Kernel data missing:
- Confirm `/kaggle/input/competitions/<slug>/` vs `/kaggle/input/<slug>/`.
- Check `kernel-metadata.json` competition and dataset sources.
- Fork a working notebook when metadata-based attachment fails.

Bad fallback logic:
- Never use training IDs/files to fake a test prediction.
- If hidden test files are unavailable in run mode, generate a valid sample-submission-shaped placeholder.

Runtime or memory failures:
- Reduce batch size, chunk inference, cache features, use quantization, or move GPU-heavy work to Kaggle kernels.

Silent failures:
- Avoid broad `except: pass`; log filename, exception type, and traceback summary.

## Final Response

Report:
- competition slug or local path
- data and metric understanding
- files created or changed
- baseline/submission status
- current score if available
- next highest-ROI experiment

$ARGUMENTS\
"""


def _kaggle_prompt(args: str) -> str:
    text = _KAGGLE_PROMPT
    if args:
        text = text.replace("$ARGUMENTS",
                            f"\n## User Competition Input\n\n{args}")
    else:
        text = text.replace("$ARGUMENTS", "")
    return text


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_bundled_skills() -> None:
    """Register all built-in skills. Called once at startup."""
    register_skill(Skill(
        name="simplify",
        description="Review changed code for reuse, quality, and efficiency, then fix issues found",
        when_to_use="After making code changes, to clean up and improve the code",
        user_invocable=True,
        argument_hint="focus",
        source="bundled",
        _prompt_fn=_simplify_prompt,
    ))

    register_skill(Skill(
        name="review",
        description="Review code changes and report issues without making fixes",
        when_to_use="To get feedback on code changes before committing",
        user_invocable=True,
        argument_hint="focus",
        source="bundled",
        _prompt_fn=_review_prompt,
    ))

    register_skill(Skill(
        name="commit",
        description="Stage changes and create a well-structured git commit",
        when_to_use="When ready to commit changes to git",
        user_invocable=True,
        argument_hint="message",
        source="bundled",
        _prompt_fn=_commit_prompt,
    ))

    register_skill(Skill(
        name="test",
        description="Run the project's test suite and analyze results",
        when_to_use="To verify code changes haven't broken anything",
        user_invocable=True,
        argument_hint="filter",
        source="bundled",
        _prompt_fn=_test_prompt,
    ))

    register_skill(Skill(
        name="kaggle",
        description="Run AutoDS's general Kaggle competition workflow",
        when_to_use="When starting or iterating on a Kaggle competition, submission, notebook, or leaderboard task",
        user_invocable=True,
        argument_hint="competition",
        source="bundled",
        _prompt_fn=_kaggle_prompt,
    ))
