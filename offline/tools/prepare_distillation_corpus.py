from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


TARGET_BUCKETS = ["1st", "10pct", "20pct", "40pct", "70pct"]
CODE_SUFFIXES = {".ipynb", ".irnb", ".py", ".R", ".Rmd"}
RANK_DIR_RE = re.compile(r"^rank(?P<rank>\d+)_(?P<bucket>1st|10pct|20pct|40pct|70pct)$")

GOOD_PATTERNS = [
    r"\bfit\s*\(",
    r"\bpredict(?:_proba)?\s*\(",
    r"\btrain_test_split\b",
    r"\bKFold\b|\bStratifiedKFold\b|\bGroupKFold\b",
    r"\bcross_val_score\b|\bcv\b",
    r"\bLightGBM\b|\blgbm?\b|\bXGBoost\b|\bxgb\b|\bCatBoost\b",
    r"\bRandomForest\b|\bLogisticRegression\b|\bRidge\b|\bLasso\b",
    r"\btorch\b|\bTensorFlow\b|\bkeras\b|\btransformers\b|\bBert\b|\bCNN\b|\bUNet\b",
    r"\bfeature\b|\bencode\b|\bembedding\b|\baugmentation\b|\bensemble\b|\bblend\b",
    r"\bmetric\b|\bscore\b|\bauc\b|\brmse\b|\bf1\b|\baccuracy\b|\bloss\b",
    r"\bsubmission\b|\bto_csv\b",
]

BAD_NAME_PATTERNS = [
    "random",
    "sample-submission",
    "sample_submission",
    "download",
    "downloader",
    "google-drive",
    "error-analysis",
    "write-up",
    "eda-only",
]

BAD_TEXT_PATTERNS = [
    r"\brandom\s+guess\b",
    r"\ball\s+random\b",
    r"\bdownload\s+datasets?\b",
    r"\bdownload\s+images?\b",
    r"\bgoogle\s+drive\b",
    r"\bwrite[- ]?up\b",
]


@dataclass
class FileScore:
    domain: str
    competition: str
    bucket: str
    rank: int
    path: str
    suffix: str
    sha256: str
    size_bytes: int
    code_chars: int
    markdown_chars: int
    code_lines: int
    code_cells: int
    markdown_cells: int
    good_hits: int
    bad_hits: int
    score: int
    quality: str
    reasons: str


def text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def read_source(path: Path) -> tuple[str, str, int, int]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(raw)
        except json.JSONDecodeError:
            return raw, "", 0, 0
        code_parts: list[str] = []
        markdown_parts: list[str] = []
        code_cells = 0
        markdown_cells = 0
        for cell in nb.get("cells", []):
            source = cell.get("source", [])
            if isinstance(source, list):
                source_text = "".join(source)
            else:
                source_text = str(source)
            if cell.get("cell_type") == "code":
                code_cells += 1
                code_parts.append(source_text)
            elif cell.get("cell_type") == "markdown":
                markdown_cells += 1
                markdown_parts.append(source_text)
        return "\n".join(code_parts), "\n".join(markdown_parts), code_cells, markdown_cells
    return raw, "", 1 if raw.strip() else 0, 0


def score_file(root: Path, path: Path) -> FileScore | None:
    try:
        domain, competition, rank_dir, *_ = path.relative_to(root).parts
    except ValueError:
        return None
    match = RANK_DIR_RE.match(rank_dir)
    if not match:
        return None

    code, markdown, code_cells, markdown_cells = read_source(path)
    combined = f"{path.name}\n{code}\n{markdown}"
    lower_name = path.name.lower()
    lower_text = combined.lower()

    good_hits = sum(1 for pattern in GOOD_PATTERNS if re.search(pattern, combined, flags=re.IGNORECASE))
    bad_hits = sum(1 for marker in BAD_NAME_PATTERNS if marker in lower_name)
    bad_hits += sum(1 for pattern in BAD_TEXT_PATTERNS if re.search(pattern, lower_text, flags=re.IGNORECASE))

    code_lines = sum(1 for line in code.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    score = 0
    reasons: list[str] = []

    if code_lines >= 80:
        score += 20
    elif code_lines >= 35:
        score += 12
    elif code_lines >= 12:
        score += 4
    else:
        score -= 25
        reasons.append("too_short")

    if code_cells >= 6:
        score += 8
    elif path.suffix != ".ipynb" and code_lines >= 35:
        score += 4

    score += min(good_hits * 6, 36)
    if markdown.strip():
        score += 6
    if match.group("bucket") == "1st":
        score += 5
    if path.suffix == ".ipynb":
        score += 3

    if bad_hits:
        score -= bad_hits * 12
        reasons.append("bad_keyword")
    if "submission" in lower_name and code_lines < 60:
        score -= 15
        reasons.append("submission_only")
    if "ensemble" in lower_name or "blend" in lower_name:
        if good_hits < 4 or code_lines < 50:
            score -= 12
            reasons.append("thin_ensemble")

    if score >= 45:
        quality = "strong"
    elif score >= 25:
        quality = "usable"
    elif score >= 10:
        quality = "weak"
    else:
        quality = "reject"

    return FileScore(
        domain=domain,
        competition=competition,
        bucket=match.group("bucket"),
        rank=int(match.group("rank")),
        path=str(path.relative_to(root)),
        suffix=path.suffix,
        sha256=text_hash(code),
        size_bytes=path.stat().st_size,
        code_chars=len(code),
        markdown_chars=len(markdown),
        code_lines=code_lines,
        code_cells=code_cells,
        markdown_cells=markdown_cells,
        good_hits=good_hits,
        bad_hits=bad_hits,
        score=score,
        quality=quality,
        reasons=";".join(reasons),
    )


def choose_ready(scores: list[FileScore], target_per_domain: int) -> list[FileScore]:
    grouped: dict[tuple[str, str, str], list[FileScore]] = defaultdict(list)
    for item in scores:
        if item.quality == "reject":
            continue
        grouped[(item.domain, item.competition, item.bucket)].append(item)

    best_by_bucket: dict[tuple[str, str, str], FileScore] = {}
    seen_hashes: set[str] = set()
    for key, values in grouped.items():
        ranked = sorted(values, key=lambda x: (x.score, -abs(bucket_rank_weight(x.bucket) - x.rank), -x.size_bytes), reverse=True)
        for value in ranked:
            if value.sha256 not in seen_hashes:
                best_by_bucket[key] = value
                seen_hashes.add(value.sha256)
                break
        if key not in best_by_bucket:
            best_by_bucket[key] = ranked[0]

    comp_buckets: dict[tuple[str, str], dict[str, FileScore]] = defaultdict(dict)
    for (domain, competition, bucket), item in best_by_bucket.items():
        comp_buckets[(domain, competition)][bucket] = item

    complete_comps: list[tuple[str, str, int]] = []
    for (domain, competition), buckets in comp_buckets.items():
        if all(bucket in buckets for bucket in TARGET_BUCKETS):
            complete_comps.append((domain, competition, sum(buckets[b].score for b in TARGET_BUCKETS)))

    selected: list[FileScore] = []
    by_domain: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for item in complete_comps:
        by_domain[item[0]].append(item)
    for domain, comps in by_domain.items():
        for _, competition, _ in sorted(comps, key=lambda x: x[2], reverse=True)[:target_per_domain]:
            selected.extend(comp_buckets[(domain, competition)][bucket] for bucket in TARGET_BUCKETS)
    return sorted(selected, key=lambda x: (x.domain, x.competition, TARGET_BUCKETS.index(x.bucket)))


def choose_coverage(scores: list[FileScore], target_per_domain: int) -> list[FileScore]:
    grouped: dict[tuple[str, str, str], list[FileScore]] = defaultdict(list)
    for item in scores:
        grouped[(item.domain, item.competition, item.bucket)].append(item)

    best_by_bucket: dict[tuple[str, str, str], FileScore] = {}
    for key, values in grouped.items():
        best_by_bucket[key] = sorted(values, key=lambda x: (x.score, x.size_bytes), reverse=True)[0]

    comp_buckets: dict[tuple[str, str], dict[str, FileScore]] = defaultdict(dict)
    for (domain, competition, bucket), item in best_by_bucket.items():
        comp_buckets[(domain, competition)][bucket] = item

    by_domain: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    for (domain, competition), buckets in comp_buckets.items():
        if all(bucket in buckets for bucket in TARGET_BUCKETS):
            total_score = sum(buckets[bucket].score for bucket in TARGET_BUCKETS)
            fallback_count = sum(1 for bucket in TARGET_BUCKETS if buckets[bucket].quality == "reject")
            by_domain[domain].append((domain, competition, fallback_count, total_score))

    selected: list[FileScore] = []
    for domain, comps in by_domain.items():
        ranked = sorted(comps, key=lambda x: (x[2], -x[3], x[1]))
        for _, competition, _, _ in ranked[:target_per_domain]:
            selected.extend(comp_buckets[(domain, competition)][bucket] for bucket in TARGET_BUCKETS)
    return sorted(selected, key=lambda x: (x.domain, x.competition, TARGET_BUCKETS.index(x.bucket)))


def bucket_rank_weight(bucket: str) -> int:
    return {
        "1st": 1,
        "10pct": 10,
        "20pct": 20,
        "40pct": 40,
        "70pct": 70,
    }[bucket]


def write_csv(path: Path, rows: list[FileScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(FileScore.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="distillation")
    parser.add_argument("--target-per-domain", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    files = [
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix in CODE_SUFFIXES
        and ".cache" not in path.parts
        and "distillation" not in path.parts
    ]
    scores = [score for path in files if (score := score_file(root, path)) is not None]
    core = choose_ready(scores, args.target_per_domain)
    coverage = choose_coverage(scores, args.target_per_domain)
    core_set = {row.path for row in core}
    coverage_set = {row.path for row in coverage}
    rejected = [row for row in scores if row.path not in coverage_set]

    write_csv(out_dir / "all_scored.csv", sorted(scores, key=lambda x: (x.domain, x.competition, x.bucket, -x.score)))
    write_csv(out_dir / "core_manifest.csv", core)
    write_csv(out_dir / "coverage_manifest.csv", coverage)
    write_csv(out_dir / "distillation_ready_manifest.csv", coverage)
    write_csv(out_dir / "rejected_or_unused.csv", sorted(rejected, key=lambda x: (x.domain, x.competition, x.bucket, -x.score)))

    core_domain_counts = Counter(row.domain for row in core)
    core_comp_counts = Counter((row.domain, row.competition) for row in core)
    coverage_domain_counts = Counter(row.domain for row in coverage)
    coverage_comp_counts = Counter((row.domain, row.competition) for row in coverage)
    summary = {
        "total_files_scored": len(scores),
        "core_files": len(core),
        "core_competitions": len(core_comp_counts),
        "core_files_by_domain": dict(sorted(core_domain_counts.items())),
        "core_competitions_by_domain": dict(sorted(Counter(domain for domain, _ in core_comp_counts).items())),
        "coverage_files": len(coverage),
        "coverage_competitions": len(coverage_comp_counts),
        "coverage_files_by_domain": dict(sorted(coverage_domain_counts.items())),
        "coverage_competitions_by_domain": dict(sorted(Counter(domain for domain, _ in coverage_comp_counts).items())),
        "coverage_fallback_files": sum(1 for row in coverage if row.quality == "reject"),
        "coverage_weak_files": sum(1 for row in coverage if row.quality == "weak"),
        "coverage_unused_files": len(scores) - len(coverage_set),
        "quality_counts_all": dict(sorted(Counter(row.quality for row in scores).items())),
        "quality_counts_core": dict(sorted(Counter(row.quality for row in core).items())),
        "quality_counts_coverage": dict(sorted(Counter(row.quality for row in coverage).items())),
        "reject_reason_counts": dict(sorted(Counter(reason for row in scores for reason in row.reasons.split(";") if reason).items())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
