from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


BUCKETS = ["70pct", "40pct", "20pct", "10pct", "1st"]


KEY_PATTERNS = re.compile(
    r"("
    r"read_csv|sample_submission|to_csv|metric|score|validation|valid|KFold|Stratified|GroupKFold|TimeSeries|OOF|fold|"
    r"LightGBM|LGBM|XGBoost|XGB|CatBoost|RandomForest|Ridge|LogisticRegression|"
    r"torch|tensorflow|keras|transformers|BERT|RoBERTa|DeBERTa|EfficientNet|ResNet|UNet|CNN|timm|"
    r"groupby|rolling|shift|lag|agg\(|merge|LabelEncoder|OneHot|target.?enc|Tfidf|Tokenizer|"
    r"augment|albumentations|mixup|cutmix|TTA|pseudo|early.?stopping|scheduler|"
    r"ensemble|blend|stack|weighted|threshold|calibrat|postprocess|clip\(|rank|argsort|topk"
    r")",
    re.IGNORECASE,
)


def read_notebook_or_code(path: Path) -> list[dict[str, str]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(raw)
        except json.JSONDecodeError:
            return [{"type": "code", "source": raw}]
        cells = []
        for cell in nb.get("cells", []):
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else str(source)
            if text.strip():
                cells.append({"type": cell.get("cell_type", "unknown"), "source": text})
        return cells
    return [{"type": "code", "source": raw}]


def compact_cell(text: str, max_len: int = 1800) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    key_lines = [line for line in lines if KEY_PATTERNS.search(line)]
    if key_lines:
        body = "\n".join(key_lines[:80])
    else:
        body = "\n".join(lines[:80])
    if len(body) > max_len:
        body = body[:max_len] + "\n...[truncated]"
    return body


def summarize_file(root: Path, rel_path: str) -> dict:
    path = root / rel_path
    cells = read_notebook_or_code(path)
    code_cells = [c for c in cells if c["type"] == "code"]
    markdown_cells = [c for c in cells if c["type"] == "markdown"]
    key_cells = []
    for idx, cell in enumerate(cells):
        source = cell["source"]
        if cell["type"] == "markdown":
            if any(word in source.lower() for word in ["score", "validation", "model", "feature", "solution", "approach", "cv", "lb"]):
                key_cells.append((idx, cell))
        elif KEY_PATTERNS.search(source):
            key_cells.append((idx, cell))
    if not key_cells:
        key_cells = list(enumerate(cells[:8]))
    snippets = []
    for idx, cell in key_cells[:24]:
        snippets.append({
            "cell_index": idx,
            "cell_type": cell["type"],
            "snippet": compact_cell(cell["source"]),
        })
    return {
        "path": rel_path,
        "file_name": path.name,
        "suffix": path.suffix,
        "cell_count": len(cells),
        "code_cell_count": len(code_cells),
        "markdown_cell_count": len(markdown_cells),
        "snippets": snippets,
    }


def write_pack(root: Path, out_dir: Path, domain: str, competition: str, rows: dict[str, dict[str, str]]) -> None:
    pack = {
        "domain": domain,
        "competition": competition,
        "rank_buckets_low_to_high": BUCKETS,
        "buckets": {},
    }
    for bucket in BUCKETS:
        row = rows[bucket]
        pack["buckets"][bucket] = {
            "rank": int(row["rank"]),
            "quality": row["quality"],
            "score": int(row["score"]),
            "reasons": row["reasons"],
            "source": summarize_file(root, row["path"]),
        }
    target = out_dir / "evidence_packs" / domain / f"{competition}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        f"# {domain} / {competition}",
        "",
        "Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`",
        "",
    ]
    for bucket in BUCKETS:
        b = pack["buckets"][bucket]
        src = b["source"]
        md.extend([
            f"## {bucket} rank {b['rank']} quality {b['quality']} score {b['score']}",
            "",
            f"File: `{src['path']}`",
            f"Cells: {src['cell_count']} total, {src['code_cell_count']} code, {src['markdown_cell_count']} markdown",
            "",
        ])
        for snippet in src["snippets"][:12]:
            md.extend([
                f"### Cell {snippet['cell_index']} {snippet['cell_type']}",
                "",
                "```",
                snippet["snippet"],
                "```",
                "",
            ])
    md_target = out_dir / "evidence_packs_md" / domain / f"{competition}.md"
    md_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="self-evolution-ds-agent")
    parser.add_argument("--manifest", default="distillation/coverage_manifest.csv")
    parser.add_argument("--out-dir", default="distillation/v4")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    rows = list(csv.DictReader((root / args.manifest).open(encoding="utf-8")))
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["domain"], row["competition"])][row["bucket"]] = row
    for (domain, competition), bucket_rows in sorted(grouped.items()):
        if all(bucket in bucket_rows for bucket in BUCKETS):
            write_pack(root, out_dir, domain, competition, bucket_rows)
    summary = {
        "competitions": len(grouped),
        "bucket_files": len(rows),
        "out_dir": str(out_dir.relative_to(root)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
