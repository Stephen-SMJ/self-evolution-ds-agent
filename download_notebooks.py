from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import pandas as pd


DOMAINS = {
    "Tabular": [
        "titanic", "house-prices-advanced-regression-techniques", "spaceship-titanic",
        "santander-customer-transaction-prediction", "competitive-data-science-predict-future-sales",
        "prudential-life-insurance-assessment", "porto-seguro-safe-driver-prediction",
        "home-credit-default-risk", "bike-sharing-demand", "otto-group-product-classification-challenge",
        "forest-cover-type-prediction", "ghouls-goblins-and-ghosts-boo", "tabular-playground-series-dec-2021",
    ],
    "NLP": [
        "nlp-getting-started", "feedback-prize-english-language-learning", "commonlitreadabilityprize",
        "jigsaw-toxic-comment-classification-challenge", "quora-question-pairs",
        "sentiment-analysis-on-movie-reviews", "word2vec-nlp-tutorial", "bag-of-words-meets-bags-of-popcorn",
        "google-quest-challenge", "natural-language-processing-with-disaster-tweets",
        "jigsaw-unintended-bias-in-toxicity-classification", "tweet-sentiment-extraction",
    ],
    "CV": [
        "digit-recognizer", "dogs-vs-cats-redux-kernels-edition", "cassava-leaf-disease-classification",
        "tgs-salt-identification-challenge", "severstal-steel-defect-detection",
        "state-farm-distracted-driver-detection", "imaterialist-challenge-fashion-2018",
        "painter-by-numbers", "understanding_cloud_organization", "humpback-whale-identification",
    ],
    "Time-Series": [
        "store-sales-time-series-forecasting", "tabular-playground-series-jan-2022",
        "amp-parkinsons-disease-progression-prediction", "g-research-crypto-forecasting",
        "ubiquant-market-prediction", "m5-forecasting-accuracy",
        "walmart-recruiting-store-sales-forecasting", "rossmann-store-sales",
        "web-traffic-time-series-forecasting",
    ],
    "Audio": [
        "birdclef-2023", "birdclef-2022", "freesound-audio-tagging-2019",
        "birdclef-2021", "rfcx-species-audio-detection", "birdsong-recognition",
        "heartbeat-sounds", "birdclef-2024", "esc50-event-classification",
    ],
    "RL": [
        "kore-2022", "santa-2022", "halite", "google-football", "connectx",
        "lux-ai-2021", "hungry-geese",
    ],
    "RecSys": [
        "h-and-m-personalized-fashion-recommendations", "otto-recommender-system",
        "santander-product-recommendation", "elo-merchant-category-recommendation",
        "expedia-hotel-recommendations", "predict-west-nile-virus",
        "instacart-market-basket-analysis", "talkingdata-adtracking-fraud-detection",
        "outbrain-click-prediction", "avazu-ctr-prediction", "recruit-restaurant-visitor-forecasting",
    ],
    "GenAI": [
        "stable-diffusion-image-to-prompts", "llm-prompt-recovery",
        "llm-detect-ai-generated-text", "kaggle-llm-science-exam",
        "llms-you-cant-please-them-all", "drawing-with-llms",
        "feedback-prize-effectiveness", "feedback-prize-2021",
        "google-ai4code", "learning-equality-curriculum-recommendations",
    ],
    "Medical": [
        "rsna-breast-cancer-detection", "histopathologic-cancer-detection",
        "siim-isic-melanoma-classification", "rsna-pneumonia-detection-challenge",
        "osic-pulmonary-fibrosis-progression", "aptos2019-blindness-detection",
        "rsna-intracranial-hemorrhage-detection", "prostate-cancer-grade-assessment",
        "hubmap-kidney-segmentation-connectivity", "sartorius-cell-instance-segmentation",
        "uw-madison-gi-tract-image-segmentation", "pangea-liver-segmentation",
    ],
}

TARGETS = [
    ("1st", 0.0),
    ("10pct", 0.1),
    ("20pct", 0.2),
    ("40pct", 0.4),
    ("70pct", 0.7),
]
TARGET_BUCKETS = [name for name, _ in TARGETS]
CODE_SUFFIXES = {".ipynb", ".irnb", ".py", ".R", ".Rmd"}
RANK_DIR_RE = re.compile(r"^rank(\d+)_(1st|10pct|20pct|40pct|70pct)$")


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def kaggle_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "kaggle", *args]


def has_code_file(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() and p.suffix in CODE_SUFFIXES for p in path.iterdir())


def bucket_counts(root: Path) -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {b: 0 for b in TARGET_BUCKETS})
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in CODE_SUFFIXES:
            continue
        try:
            domain, comp, rank_dir, *_ = p.relative_to(root).parts
        except ValueError:
            continue
        match = RANK_DIR_RE.match(rank_dir)
        if match:
            counts[(domain, comp)][match.group(2)] += 1
    return counts


def complete_by_domain(root: Path) -> Counter[str]:
    counts = bucket_counts(root)
    complete = Counter()
    for (domain, _comp), buckets in counts.items():
        if all(buckets[b] > 0 for b in TARGET_BUCKETS):
            complete[domain] += 1
    return complete


def missing_buckets(root: Path, domain: str, comp: str) -> list[str]:
    comp_dir = root / domain / comp
    present = set()
    if comp_dir.is_dir():
        for rank_dir in comp_dir.iterdir():
            match = RANK_DIR_RE.match(rank_dir.name)
            if match and has_code_file(rank_dir):
                present.add(match.group(2))
    return [b for b in TARGET_BUCKETS if b not in present]


def read_leaderboard(competition: str, cache_dir: Path) -> pd.DataFrame | None:
    cache_path = cache_dir / "leaderboards" / f"{competition}.csv"
    if cache_path.is_file():
        return pd.read_csv(cache_path)

    with tempfile.TemporaryDirectory(prefix=f"kaggle-lb-{competition}-") as td:
        tmp = Path(td)
        result = run(kaggle_cmd("competitions", "leaderboard", "-c", competition, "--download"), cwd=tmp)
        if result.returncode != 0:
            print(f"  leaderboard failed: {result.stderr.strip()[:300]}")
            return None
        zip_files = list(tmp.glob("*.zip"))
        if zip_files:
            with ZipFile(zip_files[0]) as zf:
                zf.extractall(tmp)
        csv_files = sorted(tmp.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csv_files:
            print("  leaderboard failed: no csv extracted")
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(csv_files[0].read_bytes())
        return pd.read_csv(cache_path)


def read_kernels(competition: str, cache_dir: Path, page_size: int = 1000) -> pd.DataFrame | None:
    cache_path = cache_dir / "kernels" / f"{competition}.csv"
    if cache_path.is_file():
        return pd.read_csv(cache_path)

    result = run(kaggle_cmd("kernels", "list", "--competition", competition, "--csv", "--page-size", str(page_size)))
    if result.returncode != 0:
        print(f"  kernel list failed: {result.stderr.strip()[:300]}")
        return None
    if not result.stdout.strip():
        return None
    try:
        kernels = pd.read_csv(StringIO(result.stdout))
    except Exception as exc:
        print(f"  kernel list parse failed: {exc}")
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(result.stdout)
    return kernels


def normalize_user(value: object) -> str:
    return str(value).strip().lower()


def leaderboard_user_rows(lb: pd.DataFrame) -> pd.DataFrame | None:
    if "Rank" not in lb.columns:
        print("  leaderboard has no Rank column")
        return None
    member_col = None
    for candidate in ("TeamMemberUserNames", "TeamName"):
        if candidate in lb.columns:
            member_col = candidate
            break
    if member_col is None:
        print("  leaderboard has no user/team column")
        return None
    out = lb.copy()
    out[member_col] = out[member_col].fillna("").astype(str).str.split(",")
    out = out.explode(member_col)
    out["__user"] = out[member_col].map(normalize_user)
    out["Rank"] = pd.to_numeric(out["Rank"], errors="coerce")
    return out.dropna(subset=["Rank"])


def merge_kernels(lb: pd.DataFrame, kernels: pd.DataFrame | None) -> pd.DataFrame:
    if kernels is None or kernels.empty or "ref" not in kernels.columns:
        return pd.DataFrame()
    users = leaderboard_user_rows(lb)
    if users is None or users.empty:
        return pd.DataFrame()
    k = kernels.copy()
    k["__user"] = k["ref"].astype(str).str.split("/").str[0].map(normalize_user)
    merged = k.merge(users, on="__user", how="inner")
    if "Rank" in merged:
        merged["Rank"] = pd.to_numeric(merged["Rank"], errors="coerce")
        merged = merged.dropna(subset=["Rank"])
    return merged


def target_rank(total_teams: int, percentile: float) -> int:
    return max(1, int(total_teams * percentile))


def target_path(root: Path, domain: str, comp: str, rank: int, bucket: str) -> Path:
    return root / domain / comp / f"rank{rank}_{bucket}"


def pull_kernel(kernel_ref: str, path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    result = run(kaggle_cmd("kernels", "pull", kernel_ref, "-p", str(path), "-m"))
    if result.returncode != 0:
        print(f"    pull failed for {kernel_ref}: {result.stderr.strip()[:300]}")
        return False
    print(f"    pulled {kernel_ref} -> {path}")
    return has_code_file(path)


def candidate_from_merged(merged: pd.DataFrame, target: int, used_refs: set[str]) -> Iterable[tuple[str, int]]:
    if merged.empty or "ref" not in merged or "Rank" not in merged:
        return []
    frame = merged.copy()
    frame["rank_diff"] = (frame["Rank"].astype(float) - target).abs()
    for _, row in frame.sort_values(["rank_diff", "Rank"]).iterrows():
        ref = str(row["ref"])
        if ref in used_refs:
            continue
        yield ref, int(row["Rank"])


def read_user_kernels(username: str, comp: str, cache_dir: Path) -> pd.DataFrame | None:
    safe_user = re.sub(r"[^A-Za-z0-9_.-]+", "_", username)
    cache_path = cache_dir / "user-kernels" / comp / f"{safe_user}.csv"
    if cache_path.is_file():
        return pd.read_csv(cache_path)

    result = run(kaggle_cmd("kernels", "list", "--user", username, "--competition", comp, "--csv", "--page-size", "20"))
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        kernels = pd.read_csv(StringIO(result.stdout))
    except Exception:
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(result.stdout)
    return kernels


def user_kernel_candidates(
    lb: pd.DataFrame,
    comp: str,
    target: int,
    used_refs: set[str],
    cache_dir: Path,
    max_users: int,
) -> Iterable[tuple[str, int]]:
    users = leaderboard_user_rows(lb)
    if users is None or users.empty:
        return []
    frame = users.copy()
    frame["rank_diff"] = (frame["Rank"].astype(float) - target).abs()
    for _, row in frame.sort_values(["rank_diff", "Rank"]).head(max_users).iterrows():
        username = str(row["__user"])
        rank = int(row["Rank"])
        if not username:
            continue
        kernels = read_user_kernels(username, comp, cache_dir)
        if kernels is None:
            continue
        if "ref" not in kernels:
            continue
        for ref in kernels["ref"].astype(str):
            if ref in used_refs:
                continue
            yield ref, rank


def fill_competition(root: Path, domain: str, comp: str, args: argparse.Namespace) -> bool:
    missing = missing_buckets(root, domain, comp)
    if not missing:
        return True

    print(f"\n[{domain}] {comp}: missing {missing}")
    cache_dir = root / ".cache" / "kaggle"
    lb = read_leaderboard(comp, cache_dir)
    if lb is None or lb.empty:
        return False
    kernels = read_kernels(comp, cache_dir)
    merged = merge_kernels(lb, kernels)
    total_teams = len(lb)
    used_refs: set[str] = set()

    for bucket, percentile in TARGETS:
        if bucket not in missing:
            continue
        target = target_rank(total_teams, percentile)
        print(f"  target {bucket}: rank~{target}")

        success = False
        candidates = list(candidate_from_merged(merged, target, used_refs))[: args.max_merged_candidates]
        candidates.extend(user_kernel_candidates(lb, comp, target, used_refs, cache_dir, args.max_user_candidates))
        seen = set()

        for kernel_ref, actual_rank in candidates:
            if kernel_ref in seen:
                continue
            seen.add(kernel_ref)
            used_refs.add(kernel_ref)
            path = target_path(root, domain, comp, actual_rank, bucket)
            if has_code_file(path):
                print(f"    already has code: {path}")
                success = True
                break
            if pull_kernel(kernel_ref, path):
                success = True
                break
            if len(seen) >= args.max_candidates_per_bucket:
                break

        if not success:
            print(f"    unresolved {comp} {bucket}")

    return not missing_buckets(root, domain, comp)


def competition_order(root: Path, domain: str, competitions: list[str]) -> list[str]:
    def sort_key(comp: str) -> tuple[int, int, str]:
        missing = missing_buckets(root, domain, comp)
        existing = 5 - len(missing)
        return (len(missing), -existing, comp)
    return sorted(competitions, key=sort_key)


def print_summary(root: Path) -> None:
    complete = complete_by_domain(root)
    print("\n=== Complete competitions by domain ===")
    for domain in sorted(DOMAINS):
        print(f"{domain}: {complete.get(domain, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Corpus root directory")
    parser.add_argument("--target-per-domain", type=int, default=5)
    parser.add_argument("--max-candidates-per-bucket", type=int, default=15)
    parser.add_argument("--max-merged-candidates", type=int, default=12)
    parser.add_argument("--max-user-candidates", type=int, default=12)
    parser.add_argument("--domains", nargs="*", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    domains = args.domains or list(DOMAINS)
    print_summary(root)

    for domain in domains:
        complete = complete_by_domain(root)
        if complete.get(domain, 0) >= args.target_per_domain:
            print(f"\n=== {domain}: already satisfied ({complete.get(domain, 0)}) ===")
            continue
        print(f"\n=== Processing {domain}: {complete.get(domain, 0)}/{args.target_per_domain} complete ===")
        for comp in competition_order(root, domain, DOMAINS[domain]):
            if complete_by_domain(root).get(domain, 0) >= args.target_per_domain:
                break
            fill_competition(root, domain, comp, args)

    print_summary(root)


if __name__ == "__main__":
    main()
