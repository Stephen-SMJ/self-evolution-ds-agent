from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTODS_SRC = REPO_ROOT / "src"
if AUTODS_SRC.is_dir():
    sys.path.insert(0, str(AUTODS_SRC))

from core.config import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL  # noqa: E402
from core.llm import LLMClient  # noqa: E402


BUCKETS_LOW_TO_HIGH = ["70pct", "40pct", "20pct", "10pct", "1st"]
BUCKETS_MANIFEST = ["1st", "10pct", "20pct", "40pct", "70pct"]

SIGNAL_PATTERNS = [
    r"StratifiedKFold|GroupKFold|KFold|TimeSeriesSplit|OOF|out.of.fold|cross.?val",
    r"LightGBM|LGBM|XGBoost|XGB|CatBoost|RandomForest|Ridge|LogisticRegression",
    r"transformers|BERT|RoBERTa|DeBERTa|AutoModel|AutoTokenizer|Llama|Gemma",
    r"torch|tensorflow|keras|timm|EfficientNet|ResNet|UNet|CNN|segmentation",
    r"groupby|rolling|lag|shift|target.?enc|LabelEncoder|OneHotEncoder|TfidfVectorizer",
    r"augment|mixup|cutmix|TTA|pseudo.?label|scheduler|early.?stopping|focal",
    r"ensemble|blend|stack|weighted|rank.?average|threshold|calibrat|postprocess",
    r"sample_submission|to_csv|submission|metric|score|auc|rmse|mae|f1|accuracy|map@",
]


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_notebook_or_code(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(raw)
        except json.JSONDecodeError:
            return raw, ""
        code_parts: list[str] = []
        markdown_parts: list[str] = []
        for cell in nb.get("cells", []):
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else str(source)
            if cell.get("cell_type") == "code":
                code_parts.append(text)
            elif cell.get("cell_type") == "markdown":
                markdown_parts.append(text)
        return "\n".join(code_parts), "\n".join(markdown_parts)
    return raw, ""


def compact_excerpt(path: Path, max_chars: int = 9000) -> dict[str, Any]:
    code, markdown = read_notebook_or_code(path)
    imports = re.findall(r"^(?:import|from)\s+[^#\n]+", code, flags=re.MULTILINE)
    defs = re.findall(r"^(?:def|class)\s+[^:\n]+", code, flags=re.MULTILINE)
    signal_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) > 220:
            continue
        if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in SIGNAL_PATTERNS):
            signal_lines.append(stripped)
    markdown_lines = [
        line.strip()
        for line in markdown.splitlines()
        if line.strip().startswith("#") or any(word in line.lower() for word in ("score", "cv", "validation", "model", "solution", "feature"))
    ]
    first_code = "\n".join(code.splitlines()[:80])
    last_code = "\n".join(code.splitlines()[-80:])
    body = "\n\n".join([
        "IMPORTS:\n" + "\n".join(imports[:80]),
        "DEFS_CLASSES:\n" + "\n".join(defs[:80]),
        "MARKDOWN_SIGNALS:\n" + "\n".join(markdown_lines[:80]),
        "CODE_SIGNAL_LINES:\n" + "\n".join(signal_lines[:220]),
        "FIRST_CODE:\n" + first_code,
        "LAST_CODE:\n" + last_code,
    ])
    if len(body) > max_chars:
        body = body[:max_chars] + "\n...[truncated]"
    return {
        "path": str(path),
        "suffix": path.suffix,
        "code_chars": len(code),
        "markdown_chars": len(markdown),
        "excerpt": body,
    }


def response_text(message: Any) -> str:
    parts = []
    for block in message.content:
        if isinstance(block, dict):
            if block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
            elif "text" in block:
                parts.append(str(block["text"]))
        elif hasattr(block, "text"):
            parts.append(str(block.text))
    return "\n".join(parts).strip()


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def call_llm(client: LLMClient, model: str, system: str, prompt: str, max_tokens: int, retries: int = 3) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            msg = client.create_message(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_json(response_text(msg))
        except Exception as exc:
            last_exc = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_exc}") from last_exc


def competition_prompt(item: dict[str, Any]) -> str:
    return f"""
You are distilling Kaggle competition notebooks for a self-evolving data science agent.

Analyze one competition across five rank buckets. Your task is not to summarize code mechanically.
Infer the real upgrade path: what changed from low-rank to high-rank, why it likely improves score,
what AutoDS should learn, and what should not be copied.

Return strict JSON with this schema:
{{
  "domain": "...",
  "competition": "...",
  "bucket_summaries": {{
    "70pct": {{"approach": "...", "validation": "...", "features": "...", "models": "...", "postprocess": "...", "weaknesses": ["..."], "skills": ["..."]}},
    "40pct": {{...}},
    "20pct": {{...}},
    "10pct": {{...}},
    "1st": {{...}}
  }},
  "core_evolution": [
    {{"from": "70pct", "to": "40pct", "core_upgrade": "...", "why_it_matters": "...", "implementation_hint": "..."}},
    {{"from": "40pct", "to": "20pct", "core_upgrade": "...", "why_it_matters": "...", "implementation_hint": "..."}},
    {{"from": "20pct", "to": "10pct", "core_upgrade": "...", "why_it_matters": "...", "implementation_hint": "..."}},
    {{"from": "10pct", "to": "1st", "core_upgrade": "...", "why_it_matters": "...", "implementation_hint": "..."}}
  ],
  "distilled_gold_principles": ["...", "..."],
  "autods_recipe": {{
    "baseline": "...",
    "first_upgrades": ["..."],
    "serious_competitor_upgrades": ["..."],
    "gold_push": ["..."],
    "validation_guardrails": ["..."],
    "avoid": ["..."]
  }},
  "confidence": "high|medium|low",
  "limitations": ["..."]
}}

Competition evidence:
{json.dumps(item, ensure_ascii=False)[:52000]}
""".strip()


def domain_prompt(domain: str, competition_outputs: list[dict[str, Any]]) -> str:
    compact = [
        {
            "competition": item.get("competition"),
            "core_evolution": item.get("core_evolution", []),
            "distilled_gold_principles": item.get("distilled_gold_principles", []),
            "autods_recipe": item.get("autods_recipe", {}),
            "confidence": item.get("confidence"),
        }
        for item in competition_outputs
    ]
    return f"""
You are synthesizing domain-level skills for AutoDS from competition-level semantic distillation.

Domain: {domain}

Return strict JSON:
{{
  "domain": "{domain}",
  "recurring_upgrade_path": ["..."],
  "baseline_recipe": ["..."],
  "validation_recipe": ["..."],
  "feature_recipe": ["..."],
  "model_recipe": ["..."],
  "postprocess_and_ensemble_recipe": ["..."],
  "gold_level_principles": ["..."],
  "common_failure_modes": ["..."],
  "autods_decision_policy": ["if ..., then ..."]
}}

Competition-level distilled evidence:
{json.dumps(compact, ensure_ascii=False)[:52000]}
""".strip()


def markdown_report(competition_outputs: list[dict[str, Any]], domain_outputs: list[dict[str, Any]]) -> str:
    lines = [
        "# AutoDS Offline Distillation V3",
        "",
        "V3 is LLM-assisted semantic distillation over the V2 coverage set.",
        "",
        f"- competition traces distilled: {len(competition_outputs)}",
        f"- domain recipes distilled: {len(domain_outputs)}",
        "",
        "## Domain Recipes",
        "",
    ]
    for recipe in domain_outputs:
        lines += [f"### {recipe.get('domain')}", ""]
        for key in [
            "recurring_upgrade_path",
            "validation_recipe",
            "feature_recipe",
            "model_recipe",
            "postprocess_and_ensemble_recipe",
            "gold_level_principles",
            "common_failure_modes",
        ]:
            values = recipe.get(key, [])
            if values:
                lines.append(f"**{key}**")
                for value in values[:8]:
                    lines.append(f"- {value}")
                lines.append("")
    lines += ["## Competition Evolution Index", ""]
    for item in competition_outputs:
        lines.append(f"### {item.get('domain')} / {item.get('competition')}")
        for step in item.get("core_evolution", []):
            lines.append(f"- `{step.get('from')}` -> `{step.get('to')}`: {step.get('core_upgrade')} ({step.get('why_it_matters')})")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="self-evolution-ds-agent")
    parser.add_argument("--manifest", default="distillation/coverage_manifest.csv")
    parser.add_argument("--v2-traces", default="distillation/v2/competition_evolution_traces.json")
    parser.add_argument("--out-dir", default="distillation/v3")
    parser.add_argument("--model", default=os.getenv("AUTODS_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-domain", default=None)
    parser.add_argument("--skip-domain-synthesis", action="store_true")
    args = parser.parse_args()

    load_env_file(REPO_ROOT / "AutoDS" / ".env")
    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    comp_dir = out_dir / "competition_traces"
    domain_dir = out_dir / "domain_recipes"
    comp_dir.mkdir(parents=True, exist_ok=True)
    domain_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("AUTODS_API_KEY") or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY
    base_url = os.getenv("AUTODS_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    client = LLMClient(provider="openai", api_key=api_key, base_url=base_url)

    manifest_rows = list(csv.DictReader((root / args.manifest).open(encoding="utf-8")))
    rows_by_comp: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in manifest_rows:
        if args.only_domain and row["domain"] != args.only_domain:
            continue
        rows_by_comp.setdefault((row["domain"], row["competition"]), {})[row["bucket"]] = row

    v2_by_comp = {}
    v2_path = root / args.v2_traces
    if v2_path.is_file():
        for trace in json.loads(v2_path.read_text(encoding="utf-8")):
            v2_by_comp[(trace["domain"], trace["competition"])] = trace

    system = "You are a senior Kaggle grandmaster and data science agent distillation researcher. Return only strict JSON."
    competition_outputs: list[dict[str, Any]] = []
    items = sorted(rows_by_comp.items())
    if args.limit:
        items = items[: args.limit]

    for idx, ((domain, competition), bucket_rows) in enumerate(items, start=1):
        cache_path = comp_dir / domain / f"{competition}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file():
            output = json.loads(cache_path.read_text(encoding="utf-8"))
            competition_outputs.append(output)
            print(f"[{idx}/{len(items)}] cached {domain}/{competition}")
            continue

        evidence = {
            "domain": domain,
            "competition": competition,
            "v2_trace": v2_by_comp.get((domain, competition), {}),
            "buckets": {},
        }
        for bucket in BUCKETS_LOW_TO_HIGH:
            row = bucket_rows[bucket]
            excerpt = compact_excerpt(root / row["path"])
            evidence["buckets"][bucket] = {
                "rank": row["rank"],
                "quality": row["quality"],
                "score": row["score"],
                "path": row["path"],
                "excerpt": excerpt,
            }

        print(f"[{idx}/{len(items)}] distilling {domain}/{competition}")
        output = call_llm(client, args.model, system, competition_prompt(evidence), max_tokens=5000)
        output.setdefault("domain", domain)
        output.setdefault("competition", competition)
        cache_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        competition_outputs.append(output)

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for output in competition_outputs:
        by_domain.setdefault(output["domain"], []).append(output)

    domain_outputs: list[dict[str, Any]] = []
    if not args.skip_domain_synthesis:
        for domain, outputs in sorted(by_domain.items()):
            cache_path = domain_dir / f"{domain}.json"
            if cache_path.is_file():
                domain_output = json.loads(cache_path.read_text(encoding="utf-8"))
                domain_outputs.append(domain_output)
                print(f"cached domain {domain}")
                continue
            print(f"synthesizing domain {domain}")
            domain_output = call_llm(client, args.model, system, domain_prompt(domain, outputs), max_tokens=5000)
            domain_output.setdefault("domain", domain)
            cache_path.write_text(json.dumps(domain_output, indent=2, ensure_ascii=False), encoding="utf-8")
            domain_outputs.append(domain_output)

    summary = {
        "model": args.model,
        "competition_traces": len(competition_outputs),
        "domain_recipes": len(domain_outputs),
        "out_dir": str(out_dir.relative_to(root)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if domain_outputs:
        (out_dir / "semantic_distillation_v3.md").write_text(markdown_report(competition_outputs, domain_outputs), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
