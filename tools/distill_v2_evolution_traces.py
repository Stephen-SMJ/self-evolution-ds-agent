from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


BUCKET_ORDER = ["70pct", "40pct", "20pct", "10pct", "1st"]
OUTPUT_BUCKET_ORDER = ["1st", "10pct", "20pct", "40pct", "70pct"]


PATTERNS: dict[str, dict[str, list[str]]] = {
    "validation": {
        "train_valid_split": [r"train_test_split", r"valid(?:ation)?_split"],
        "kfold": [r"\bKFold\b", r"StratifiedKFold", r"GroupKFold", r"MultilabelStratifiedKFold"],
        "time_split": [r"TimeSeriesSplit", r"walk.forward", r"expanding.window", r"validation.*date", r"cutoff"],
        "oof": [r"\bOOF\b", r"out.of.fold", r"oof_", r"valid_preds", r"fold_preds"],
        "metric_eval": [r"roc_auc", r"accuracy_score", r"f1_score", r"mean_squared_error", r"rmse", r"mae", r"log_loss", r"map@", r"ndcg", r"score"],
    },
    "features": {
        "missing": [r"fillna", r"isnull", r"isna", r"missing", r"SimpleImputer"],
        "categorical": [r"LabelEncoder", r"OneHotEncoder", r"get_dummies", r"Categorical", r"category", r"cat_features"],
        "target_encoding": [r"target.?enc", r"TargetEncoder", r"mean.?enc", r"leaveoneout"],
        "aggregations": [r"groupby", r"agg\(", r"pivot_table", r"transform\(", r"rolling", r"expanding"],
        "text_features": [r"tokenizer", r"TF.?IDF", r"TfidfVectorizer", r"CountVectorizer", r"sentence", r"embedding"],
        "image_features": [r"albumentations", r"torchvision\.transforms", r"ImageDataGenerator", r"\bcv2\b", r"\bPIL\b", r"resize\s*\(", r"random.?crop", r"center.?crop", r"mask.?rcnn", r"segmentation"],
        "audio_features": [r"mel", r"spectrogram", r"librosa", r"torchaudio", r"mfcc", r"stft"],
        "time_features": [r"lag", r"rolling", r"shift\(", r"date", r"month", r"weekday", r"holiday"],
        "recsys_features": [r"candidate generation", r"candidate", r"co.?visitation", r"\bsession\b", r"ranker", r"recall@?", r"item.?item", r"user.?item"],
    },
    "models": {
        "linear": [r"LogisticRegression", r"Ridge", r"Lasso", r"LinearRegression", r"SGDClassifier"],
        "tree": [r"RandomForest", r"ExtraTrees", r"DecisionTree"],
        "gbm": [r"lightgbm", r"\blgb\b", r"LGBM", r"xgboost", r"\bxgb\b", r"XGB", r"catboost", r"CatBoost"],
        "cnn": [r"ResNet", r"EfficientNet", r"DenseNet", r"Conv2D", r"\bCNN\b", r"timm", r"segmentation_models"],
        "transformer": [r"transformers", r"Bert", r"RoBERTa", r"DeBERTa", r"AutoModel", r"AutoTokenizer", r"LLM", r"Gemma", r"Llama"],
        "sequence": [r"LSTM", r"GRU", r"RNN", r"TransformerEncoder"],
        "rl_search": [r"self.?play", r"simulation", r"minimax", r"MCTS", r"\bpolicy\b", r"gym\.|make\("],
    },
    "training": {
        "early_stopping": [r"early_stopping", r"EarlyStopping", r"early_stopping_rounds"],
        "scheduler": [r"scheduler", r"CosineAnnealing", r"ReduceLROnPlateau", r"warmup"],
        "class_imbalance": [r"class_weight", r"pos_weight", r"WeightedRandomSampler", r"focal", r"imbalance"],
        "augmentation": [r"augment", r"mixup", r"cutmix", r"tta", r"noise", r"flip", r"rotate", r"SpecAugment"],
        "pseudo_label": [r"pseudo.?label", r"self.training"],
    },
    "postprocess": {
        "thresholding": [r"threshold", r"argmax", r"clip\(", r"round\(", r"calibrat"],
        "ranking": [r"rankdata", r"argsort", r"topk", r"top_k", r"MAP@K", r"ndcg"],
        "submission_format": [r"sample_submission", r"to_csv", r"submission\.csv", r"zipfile"],
        "ensemble": [r"ensemble", r"blend", r"stack", r"weighted", r"averag", r"seed", r"fold.*ensemble"],
    },
    "risk": {
        "downloader_only": [r"download.*image", r"download.*dataset", r"google.?drive", r"kaggle datasets download"],
        "random_baseline": [r"random.?guess", r"np\.random", r"all random", r"sample submission all random"],
        "too_short": [r"__TOO_SHORT__"],
    },
}


@dataclass
class BucketTrace:
    bucket: str
    rank: int
    path: str
    quality: str
    score: int
    code_lines: int
    signals: dict[str, list[str]]
    dominant_patterns: list[str]
    summary: str
    weaknesses: list[str]


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


def extract_signals(text: str, code_lines: int) -> dict[str, list[str]]:
    augmented = text
    if code_lines < 20:
        augmented += "\n__TOO_SHORT__"
    signals: dict[str, list[str]] = {}
    for category, items in PATTERNS.items():
        hits: list[str] = []
        for name, patterns in items.items():
            if any(re.search(pattern, augmented, flags=re.IGNORECASE) for pattern in patterns):
                hits.append(name)
        if hits:
            signals[category] = hits
    return signals


def summarize_bucket(row: dict[str, str], signals: dict[str, list[str]]) -> tuple[str, list[str], list[str]]:
    parts: list[str] = []
    weaknesses: list[str] = []
    dominant: list[str] = []
    for category in ["validation", "features", "models", "training", "postprocess"]:
        values = signals.get(category, [])
        if values:
            dominant.extend(f"{category}:{value}" for value in values[:3])
            parts.append(f"{category}={','.join(values[:4])}")
    risks = signals.get("risk", [])
    if risks:
        weaknesses.extend(risks)
    if row["quality"] in {"weak", "reject"}:
        weaknesses.append(f"quality:{row['quality']}")
    if not signals.get("validation"):
        weaknesses.append("no_explicit_validation")
    if not signals.get("models"):
        weaknesses.append("no_clear_model")
    if not parts:
        parts.append("limited extractable modeling signal")
    return "; ".join(parts), dominant[:10], sorted(set(weaknesses))


def infer_upgrade(prev: BucketTrace, cur: BucketTrace) -> list[str]:
    upgrades: list[str] = []
    for category in ["validation", "features", "models", "training", "postprocess"]:
        prev_values = set(prev.signals.get(category, []))
        cur_values = set(cur.signals.get(category, []))
        added = sorted(cur_values - prev_values)
        if added:
            upgrades.append(f"add {category}: {', '.join(added)}")
    if cur.score > prev.score + 10:
        upgrades.append("higher-quality implementation signal")
    if prev.weaknesses and len(cur.weaknesses) < len(prev.weaknesses):
        upgrades.append("fewer detected weaknesses")
    return upgrades or ["no clear structural upgrade detected; inspect notebook semantics manually"]


def domain_lesson(domain: str, counter: Counter[str]) -> list[str]:
    common = [name for name, _ in counter.most_common(12)]
    lessons: list[str] = []
    if any("validation:" in item for item in common):
        lessons.append("Start by matching validation to the competition split risk.")
    if any("models:gbm" in item for item in common):
        lessons.append("Use GBM models as the first serious model family when structured features exist.")
    if any("models:cnn" in item for item in common):
        lessons.append("Use pretrained CNN/vision backbones before custom architectures.")
    if any("models:transformer" in item for item in common):
        lessons.append("Use pretrained transformer representations and keep a simple baseline for sanity.")
    if any("postprocess:ensemble" in item for item in common):
        lessons.append("Add controlled ensembles after individual models are validated.")
    if any("features:aggregations" in item for item in common):
        lessons.append("Prioritize group/entity/time aggregations as high-ROI features.")
    if any("features:audio_features" in item for item in common):
        lessons.append("Standardize waveform-to-spectrogram preprocessing and clip aggregation.")
    if any("features:recsys_features" in item for item in common):
        lessons.append("Separate candidate generation from ranking and measure candidate recall.")
    if domain == "RL":
        lessons.append("Build a local simulator/evaluation harness before optimizing policy logic.")
    return lessons[:8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="self-evolution-ds-agent")
    parser.add_argument("--manifest", default="distillation/coverage_manifest.csv")
    parser.add_argument("--out-dir", default="distillation/v2")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest_path = root / args.manifest
    out_dir = root / args.out_dir
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["competition"])].append(row)

    competition_traces: list[dict[str, Any]] = []
    domain_counters: dict[str, Counter[str]] = defaultdict(Counter)
    domain_quality: dict[str, Counter[str]] = defaultdict(Counter)

    for (domain, competition), comp_rows in sorted(grouped.items()):
        by_bucket = {row["bucket"]: row for row in comp_rows}
        traces: dict[str, BucketTrace] = {}
        for bucket in OUTPUT_BUCKET_ORDER:
            row = by_bucket[bucket]
            path = root / row["path"]
            code, markdown = read_notebook_or_code(path)
            code_lines = int(row["code_lines"])
            signals = extract_signals(f"{path.name}\n{code}\n{markdown}", code_lines)
            summary, dominant, weaknesses = summarize_bucket(row, signals)
            trace = BucketTrace(
                bucket=bucket,
                rank=int(row["rank"]),
                path=row["path"],
                quality=row["quality"],
                score=int(row["score"]),
                code_lines=code_lines,
                signals=signals,
                dominant_patterns=dominant,
                summary=summary,
                weaknesses=weaknesses,
            )
            traces[bucket] = trace
            domain_quality[domain][row["quality"]] += 1
            for pattern in dominant:
                domain_counters[domain][pattern] += 1

        transitions = []
        for prev_bucket, cur_bucket in zip(BUCKET_ORDER, BUCKET_ORDER[1:]):
            transitions.append({
                "from": prev_bucket,
                "to": cur_bucket,
                "upgrade_hypotheses": infer_upgrade(traces[prev_bucket], traces[cur_bucket]),
            })

        competition_traces.append({
            "domain": domain,
            "competition": competition,
            "buckets": {bucket: asdict(traces[bucket]) for bucket in OUTPUT_BUCKET_ORDER},
            "transitions_low_to_high": transitions,
            "recommended_next_autods_actions": [
                "replicate the strongest validation idea first",
                "extract one feature/model/postprocess idea at a time",
                "compare local CV and leaderboard before ensembling",
            ],
        })

    domain_recipes = []
    for domain in sorted(domain_counters):
        domain_recipes.append({
            "domain": domain,
            "quality_counts": dict(domain_quality[domain]),
            "top_patterns": domain_counters[domain].most_common(20),
            "distilled_lessons": domain_lesson(domain, domain_counters[domain]),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "competition_evolution_traces.json").write_text(
        json.dumps(competition_traces, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "domain_recipes.json").write_text(
        json.dumps(domain_recipes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "manifest": str(manifest_path.relative_to(root)),
        "competitions": len(competition_traces),
        "bucket_files": len(rows),
        "domains": sorted(domain_counters),
        "files": {
            "competition_evolution_traces": str((out_dir / "competition_evolution_traces.json").relative_to(root)),
            "domain_recipes": str((out_dir / "domain_recipes.json").relative_to(root)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
