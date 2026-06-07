# CitOrigin Custom Skill Example

This example shows how to package a project-specific autods custom skill for a
claim-evidence auditing workflow.

It is designed for the
[CitOrigin](https://github.com/meijilin/TrustWorthyOfMAS/tree/main/CitOrigin)
repository, not as a built-in autods skill.

## What it does

The `citorigin` skill helps an agent:

- score externally provided claims against evidence blocks or documents
- choose between `shapley` and `drop_hold_per_unit`
- generate an HTML result page for inspection
- inspect or summarize result JSON files

## Install

Copy the skill into one of autods's discovery locations:

- project-local: `<project>/.autods/skills/citorigin/SKILL.md`
- user-global: `~/.autods/skills/citorigin/SKILL.md`

Example for a project-local install:

```bash
mkdir -p .autods/skills/citorigin
cp docs/examples/citorigin/SKILL.md .autods/skills/citorigin/SKILL.md
```

## Repository assumptions

This skill assumes:

- the current working directory is the CitOrigin repository root
- `python` resolves to an environment where CitOrigin dependencies are installed
- CLI commands are run with `PYTHONPATH=src`

The example intentionally avoids machine-specific absolute paths.

## Example usage

```text
/citorigin run the main workflow on one claim and generate an HTML viewer
```

Example tasks this skill is designed to handle:

- score one JSON payload with `score-claim`
- score one claim plus PDF or TXT evidence with `score-claim-from-files`
- score one example directory of claims and documents with `score-claims-from-example`
- build an HTML viewer from the result JSON

## Why this is an example, not a built-in skill

`citorigin` is a domain-specific workflow tied to one repository layout and one
set of CLI commands. It fits autods's custom skill mechanism well, but it is
not broad enough to be a bundled built-in skill like `/review` or `/test`.
