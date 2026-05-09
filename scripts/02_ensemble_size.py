import os
import csv
import itertools
import json
import math
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError(
        "python-dotenv is required. Install the conda environment from environment.yml."
    ) from exc

# ── Cache / Auth ──────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

OUT_DIR = os.getenv("OUTPUT_DIR", "./majority_regime_analysis_results")
os.makedirs(OUT_DIR, exist_ok=True)


import numpy as np
from datasets import load_dataset


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAMES = [
    # "ibm-granite/granite-3.3-2b-instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-270m-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-3-1b-it",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

SHORT_NAMES = [m.split("/")[-1] for m in MODEL_NAMES]

BENCHMARK = "mmlu"
N_SAMPLES = 14042 # 14042 9981 546 1274
DATASET_SEED = 42
DATASET_K_FILTER: Union[str, int] = "modal"
INCLUDE_PASSAGE_IN_QUESTION = True
RACE_CONFIG = "all"
GPQA_SHUFFLE_OPTIONS = True

OUT_DIR = "./majority_regime_analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-6

MAX_CYCLIC_SHIFTS = None
CYCLIC_SHIFT_SEED = 123

CACHE_TAG = f"{BENCHMARK}_K{DATASET_K_FILTER}_N{N_SAMPLES}_seed{DATASET_SEED}_M{len(MODEL_NAMES)}"
CYCLIC_TAG = (
    f"{CACHE_TAG}_cyclic"
    f"_S{'all' if MAX_CYCLIC_SHIFTS is None else MAX_CYCLIC_SHIFTS}"
    f"_shiftseed{CYCLIC_SHIFT_SEED}"
)
CYCLIC_NPZ_FILE = os.path.join(OUT_DIR, f"cyclic_gen_scores_{CYCLIC_TAG}.npz")

JSON_FILE = os.path.join(OUT_DIR, f"ensemble_size_ablation_{CACHE_TAG}.json")
ROWS_CSV_FILE = os.path.join(OUT_DIR, f"ensemble_size_ablation_rows_{CACHE_TAG}.csv")
SUMMARY_CSV_FILE = os.path.join(OUT_DIR, f"ensemble_size_ablation_summary_{CACHE_TAG}.csv")
HETEROGENEITY_CORR_CSV_FILE = os.path.join(OUT_DIR, f"ensemble_size_heterogeneity_corr_{CACHE_TAG}.csv")
HETEROGENEITY_BIN_RANGE_CSV_FILE = os.path.join(OUT_DIR, f"ensemble_size_heterogeneity_bins_range_{CACHE_TAG}.csv")
HETEROGENEITY_BIN_STD_CSV_FILE = os.path.join(OUT_DIR, f"ensemble_size_heterogeneity_bins_std_{CACHE_TAG}.csv")
SIGNAL_VALIDITY_CSV_FILE = os.path.join(
    OUT_DIR, f"signal_validity_model_question_{CACHE_TAG}.csv"
)
SIGNAL_VALIDITY_MODEL_CSV_FILE = os.path.join(
    OUT_DIR, f"signal_validity_by_model_{CACHE_TAG}.csv"
)

ORDINAL_AGG_METHODS = ["hard_majority", "borda", "mrr", "irv"]

RBO_P = 0.85
STABILITY_TEMPERATURE = 1.0

# Set to an int for debugging, e.g. 200.
MAX_SUBENSEMBLES_PER_SIZE = None
SUBENSEMBLE_SAMPLE_SEED = 12345

# McNemar is relatively cheap, but not needed for the ensemble-size figure.
# Set True if you want exact p-values per individual subset row.
COMPUTE_ROW_MCNEMAR = False

HETEROGENEITY_N_BINS = 3
HETEROGENEITY_PRIMARY_METHOD = "mrr"


# =============================================================================
# Dataset
# =============================================================================

@dataclass
class DatasetPack:
    questions: List[str]
    options: List[List[str]]
    correct: np.ndarray
    option_labels: List[str]
    source_indices: List[int]
    question_ids: List[str]
    benchmark_resolved: str
    k_filter: str


def stable_json_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def canonical_benchmark_name(benchmark):
    b = benchmark.strip().lower().replace("-", "_")
    aliases = {
        "mmlupro": "mmlu_pro",
        "mmlu_pro": "mmlu_pro",
        "mmlu": "mmlu",
        "arc_easy": "arc_easy",
        "arceasy": "arc_easy",
        "arc_e": "arc_easy",
        "arc_challenge": "arc_challenge",
        "arcchallenge": "arc_challenge",
        "arc_c": "arc_challenge",
        "race": "race",
        "race_all": "race",
        "race_high": "race_high",
        "race_middle": "race_middle",
        "gpqa": "gpqa_diamond",
        "gpqa_diamond": "gpqa_diamond",
        "gpqa_main": "gpqa_main",
        "gpqa_extended": "gpqa_extended",
        "gpqa_experts": "gpqa_experts",
        "medqa": "medqa",
        "openlifescienceai_medqa": "medqa",
    }
    if b not in aliases:
        raise ValueError(f"Unknown BENCHMARK={benchmark!r}")
    return aliases[b]

def compute_signal_validity_rows(rbo_qm, pred_qm, correct, short_names=None):
    """
    Tests whether phi_m(q) tracks per-question correctness.

    Inputs:
      rbo_qm:   Q x M stability signal phi_m(q)
      pred_qm:  Q x M single-model cyclic-mean predictions
      correct:  Q correct labels

    Returns:
      summary: global summary with X, Y, Z, p-value, r_min, r_max
      model_rows: one row per model with within-model correlations
      mq_rows: optional model-question level rows for CSV export
    """
    rbo_qm = np.asarray(rbo_qm, dtype=np.float64)
    pred_qm = np.asarray(pred_qm, dtype=np.int64)
    correct = np.asarray(correct, dtype=np.int64)

    Q, M = rbo_qm.shape

    if pred_qm.shape != (Q, M):
        raise ValueError(f"pred_qm shape mismatch: got {pred_qm.shape}, expected {(Q, M)}")

    correctness_qm = (pred_qm == correct[:, None]).astype(np.int64)

    phi_flat = rbo_qm.reshape(-1)
    correct_flat = correctness_qm.reshape(-1)

    finite = np.isfinite(phi_flat) & np.isfinite(correct_flat)
    phi_flat = phi_flat[finite]
    correct_flat = correct_flat[finite]

    phi_correct = phi_flat[correct_flat == 1]
    phi_wrong = phi_flat[correct_flat == 0]

    X = float(np.mean(phi_correct)) if len(phi_correct) else float("nan")
    Y = float(np.mean(phi_wrong)) if len(phi_wrong) else float("nan")

    # Point-biserial correlation is Pearson correlation with a binary variable.
    Z = safe_corr(phi_flat, correct_flat)

    try:
        from scipy.stats import pointbiserialr
        Z_scipy, p_value = pointbiserialr(correct_flat.astype(int), phi_flat)
        Z = float(Z_scipy)
        p_value = float(p_value)
    except Exception:
        p_value = float("nan")

    model_rows = []
    within_rs = []

    for mi in range(M):
        phi_m = rbo_qm[:, mi]
        corr_m = correctness_qm[:, mi].astype(np.int64)

        mask = np.isfinite(phi_m) & np.isfinite(corr_m)
        phi_m = phi_m[mask]
        corr_m = corr_m[mask]

        r_m = safe_corr(phi_m, corr_m)

        try:
            from scipy.stats import pointbiserialr
            if len(np.unique(corr_m)) >= 2 and np.std(phi_m) > EPS:
                r_scipy, p_m = pointbiserialr(corr_m.astype(int), phi_m)
                r_m = float(r_scipy)
                p_m = float(p_m)
            else:
                p_m = float("nan")
        except Exception:
            p_m = float("nan")

        if np.isfinite(r_m):
            within_rs.append(r_m)

        name = short_names[mi] if short_names is not None else f"model_{mi}"

        model_rows.append({
            "model_index": int(mi),
            "model": name,
            "n_questions": int(len(phi_m)),
            "accuracy": float(np.mean(corr_m)) if len(corr_m) else float("nan"),
            "phi_mean": float(np.mean(phi_m)) if len(phi_m) else float("nan"),
            "phi_correct_mean": float(np.mean(phi_m[corr_m == 1])) if np.any(corr_m == 1) else float("nan"),
            "phi_wrong_mean": float(np.mean(phi_m[corr_m == 0])) if np.any(corr_m == 0) else float("nan"),
            "phi_correct_minus_wrong": (
                float(np.mean(phi_m[corr_m == 1]) - np.mean(phi_m[corr_m == 0]))
                if np.any(corr_m == 1) and np.any(corr_m == 0)
                else float("nan")
            ),
            "within_model_r": r_m,
            "within_model_p": p_m,
        })

    r_min = float(np.min(within_rs)) if within_rs else float("nan")
    r_max = float(np.max(within_rs)) if within_rs else float("nan")

    # Optional: pooled within-model correlation after removing model-level means.
    # This is a useful robustness check against average model competence.
    phi_centered = np.zeros_like(rbo_qm, dtype=np.float64)
    corr_centered = np.zeros_like(correctness_qm, dtype=np.float64)

    for mi in range(M):
        phi_centered[:, mi] = rbo_qm[:, mi] - np.mean(rbo_qm[:, mi])
        corr_centered[:, mi] = correctness_qm[:, mi] - np.mean(correctness_qm[:, mi])

    pooled_within_r = safe_corr(phi_centered.reshape(-1), corr_centered.reshape(-1))

    mq_rows = []
    for qi in range(Q):
        for mi in range(M):
            name = short_names[mi] if short_names is not None else f"model_{mi}"
            mq_rows.append({
                "question_index": int(qi),
                "model_index": int(mi),
                "model": name,
                "phi": float(rbo_qm[qi, mi]),
                "correct": int(correctness_qm[qi, mi]),
                "prediction": int(pred_qm[qi, mi]),
                "gold": int(correct[qi]),
            })

    summary = {
        "n_questions": int(Q),
        "n_models": int(M),
        "n_model_question_predictions": int(Q * M),
        "phi_correct_mean": X,
        "phi_wrong_mean": Y,
        "phi_correct_minus_wrong": float(X - Y),
        "point_biserial_r": Z,
        "point_biserial_p": p_value,
        "within_model_r_min": r_min,
        "within_model_r_max": r_max,
        "pooled_within_model_centered_r": pooled_within_r,
    }

    return summary, model_rows, mq_rows

def choose_target_k(option_lengths, benchmark):
    if isinstance(DATASET_K_FILTER, int):
        return int(DATASET_K_FILTER), f"explicit_{int(DATASET_K_FILTER)}"

    mode = str(DATASET_K_FILTER).lower()

    if mode == "max":
        return int(max(option_lengths)), "max"

    if mode == "modal":
        vals, counts = np.unique(np.asarray(option_lengths, dtype=np.int64), return_counts=True)
        return int(vals[np.argmax(counts)]), "modal"

    if mode == "none":
        unique = sorted(set(int(x) for x in option_lengths))
        if len(unique) != 1:
            raise ValueError(f"DATASET_K_FILTER='none' requires uniform K, got {unique}")
        return int(unique[0]), "none_uniform"

    if benchmark in [
        "mmlu", "race", "race_high", "race_middle",
        "gpqa_diamond", "gpqa_main", "gpqa_extended", "gpqa_experts",
        "medqa",
    ]:
        return 4, "benchmark_fixed_4"

    raise ValueError(f"Unknown DATASET_K_FILTER={DATASET_K_FILTER!r}")


def normalize_answer_label(answer, labels):
    s = str(answer).strip()
    if s in labels:
        return labels.index(s)
    if s.upper() in labels:
        return labels.index(s.upper())
    if s.isdigit():
        if s in labels:
            return labels.index(s)
        idx0 = int(s) - 1
        if 0 <= idx0 < len(labels):
            return idx0
    raise ValueError(f"Cannot map answer={answer!r} to labels={labels!r}")


def remap_to_standard_labels(options, correct_idx):
    if not (0 <= int(correct_idx) < len(options)):
        raise ValueError(f"correct_idx out of range: {correct_idx}")
    return [str(o) for o in options], int(correct_idx)


def shuffle_options_deterministic(options, correct_idx, rng):
    perm = rng.permutation(len(options))
    shuffled = [str(options[int(j)]) for j in perm]
    new_correct = int(np.where(perm == int(correct_idx))[0][0])
    return shuffled, new_correct


def get_gpqa_field(ex, candidates):
    for k in candidates:
        if k in ex and ex[k] is not None:
            return str(ex[k])
    raise KeyError(f"Could not find any of {candidates}; available={sorted(ex.keys())}")


def load_dataset_pack(n_samples, benchmark, seed):
    benchmark = canonical_benchmark_name(benchmark)
    examples = []

    if benchmark == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        for i, ex in enumerate(ds):
            examples.append((i, str(ex["question"]), [str(o) for o in ex["options"]], int(ex["answer_index"])))

    elif benchmark == "mmlu":
        ds = load_dataset("cais/mmlu", name="all", split="test")
        for i, ex in enumerate(ds):
            examples.append((i, str(ex["question"]), [str(o) for o in ex["choices"]], int(ex["answer"])))

    elif benchmark in ["arc_easy", "arc_challenge"]:
        arc_config = "ARC-Easy" if benchmark == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", arc_config, split="test")
        for i, ex in enumerate(ds):
            labels = [str(x).strip() for x in ex["choices"]["label"]]
            opts = [str(x) for x in ex["choices"]["text"]]
            try:
                correct_idx = normalize_answer_label(ex["answerKey"], labels)
            except Exception:
                ans = str(ex["answerKey"]).strip()
                if ans not in opts:
                    raise
                correct_idx = opts.index(ans)
            examples.append((i, str(ex["question"]), opts, correct_idx))

    elif benchmark in ["race", "race_high", "race_middle"]:
        race_cfg = {"race_high": "high", "race_middle": "middle"}.get(benchmark, RACE_CONFIG)
        ds = load_dataset("EleutherAI/race", race_cfg, split="test")
        for i, ex in enumerate(ds):
            article = str(ex.get("article", ""))
            q = f"Passage: {article}\n\nQuestion: {ex['question']}" if INCLUDE_PASSAGE_IN_QUESTION else str(ex["question"])
            opts = [str(o) for o in ex["options"]]
            labels = [chr(ord("A") + j) for j in range(len(opts))]
            examples.append((i, q, opts, normalize_answer_label(ex["answer"], labels)))

    elif benchmark in ["gpqa_diamond", "gpqa_main", "gpqa_extended", "gpqa_experts"]:
        ds = load_dataset("Idavidrein/gpqa", benchmark, split="train")
        for i, ex in enumerate(ds):
            question = get_gpqa_field(ex, ["Question", "question"])
            correct_ans = get_gpqa_field(ex, ["Correct Answer", "correct_answer", "correct answer"])
            inc1 = get_gpqa_field(ex, ["Incorrect Answer 1", "incorrect_answer_1", "incorrect answer 1"])
            inc2 = get_gpqa_field(ex, ["Incorrect Answer 2", "incorrect_answer_2", "incorrect answer 2"])
            inc3 = get_gpqa_field(ex, ["Incorrect Answer 3", "incorrect_answer_3", "incorrect answer 3"])
            examples.append((i, question, [correct_ans, inc1, inc2, inc3], 0))

    elif benchmark == "medqa":
        ds = load_dataset("openlifescienceai/medqa", split="test")
        for i, ex in enumerate(ds):
            d = ex["data"]
            question = str(d["Question"])
            options_dict = d["Options"]
            labels = sorted(options_dict.keys())
            opts = [str(options_dict[l]) for l in labels]
            correct_label = str(d["Correct Option"]).strip().upper()
            correct_idx = normalize_answer_label(correct_label, labels)
            examples.append((i, question, opts, correct_idx))

    else:
        raise ValueError(f"Unhandled benchmark: {benchmark}")

    option_lengths = [len(o) for _, _, o, _ in examples]
    target_k, k_filter_name = choose_target_k(option_lengths, benchmark)

    filtered = [
        (i, q, o, a)
        for (i, q, o, a) in examples
        if len(o) == target_k and 0 <= int(a) < target_k
    ]
    if not filtered:
        raise RuntimeError(f"No examples after K filter: target_k={target_k}")

    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(filtered), size=min(n_samples, len(filtered)), replace=False).tolist())

    qs, opts_all, correct, source_indices, qids = [], [], [], [], []

    for j in idxs:
        source_i, q, opts, a = filtered[int(j)]
        opts, a = remap_to_standard_labels(opts, int(a))

        if benchmark.startswith("gpqa") and GPQA_SHUFFLE_OPTIONS:
            opts, a = shuffle_options_deterministic(opts, a, rng)

        qid = stable_json_hash({
            "benchmark": benchmark,
            "source_index": int(source_i),
            "question": q,
            "options": opts,
            "correct": int(a),
            "k": int(target_k),
            "gpqa_shuffle": bool(benchmark.startswith("gpqa") and GPQA_SHUFFLE_OPTIONS),
        })

        qs.append(q)
        opts_all.append(opts)
        correct.append(a)
        source_indices.append(int(source_i))
        qids.append(qid)

    k = len(opts_all[0])
    hist = {int(v): int(np.sum(np.asarray(option_lengths) == v)) for v in sorted(set(option_lengths))}
    print(f"Loaded {len(qs)}/{len(filtered)} examples | {benchmark} | K={k} | K_filter={k_filter_name} | raw_K_hist={hist}")

    return DatasetPack(
        questions=qs,
        options=opts_all,
        correct=np.asarray(correct, dtype=np.int64),
        option_labels=[chr(ord("A") + i) for i in range(k)],
        source_indices=source_indices,
        question_ids=qids,
        benchmark_resolved=benchmark,
        k_filter=k_filter_name,
    )


# =============================================================================
# Utilities
# =============================================================================

def serialise(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialise(v) for v in obj]
    if isinstance(obj, tuple):
        return [serialise(v) for v in obj]
    return obj


def fmt_float(x, nd=4):
    try:
        if np.isnan(x):
            return "nan"
    except Exception:
        pass
    return f"{float(x):.{nd}f}"


def print_table(rows, title, headers=None, nd=4, max_rows=None):
    print(f"\n--- {title} ---")
    if not rows:
        print("<empty>")
        return

    if max_rows is not None:
        rows = rows[:max_rows]

    if headers is None:
        headers = list(rows[0].keys())

    str_rows = []
    for r in rows:
        rr = {}
        for h in headers:
            v = r.get(h, "")
            rr[h] = fmt_float(v, nd) if isinstance(v, float) else str(v)
        str_rows.append(rr)

    widths = {h: max(len(h), max(len(r[h]) for r in str_rows)) for h in headers}
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))

    for r in str_rows:
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))


def write_csv(path, rows, headers=None):
    if not rows:
        return
    if headers is None:
        headers = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def logsoftmax(x, axis):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    z = np.exp(x - m)
    z /= np.maximum(np.sum(z, axis=axis, keepdims=True), EPS)
    return np.log(np.maximum(z, EPS))


def normalize(p, axis=None):
    p = np.maximum(np.asarray(p, dtype=np.float64), EPS)
    return p / np.maximum(np.sum(p, axis=axis, keepdims=True), EPS)


def zscore_per_question(x_qm):
    x = np.asarray(x_qm, dtype=np.float64)
    mu = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True)
    return (x - mu) / np.where(sd <= EPS, 1.0, sd)


def softmax_weights(x_qm, temperature=1.0):
    z = np.asarray(x_qm, dtype=np.float64) / max(float(temperature), EPS)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    p = e / np.maximum(np.sum(e, axis=1, keepdims=True), EPS)
    return p * x_qm.shape[1]


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def mcnemar_exact_p(wins, losses):
    wins = int(wins)
    losses = int(losses)
    n = wins + losses
    if n == 0:
        return float("nan")

    k = min(wins, losses)
    try:
        from scipy.stats import binom
        return float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))
    except Exception:
        pass

    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(k + 1)
    ]
    m = max(log_terms)
    cdf = math.exp(m) * sum(math.exp(t - m) for t in log_terms)
    return float(min(1.0, 2.0 * cdf))


# =============================================================================
# Cache loading
# =============================================================================

def choose_cyclic_shifts(K, max_shifts, seed):
    if max_shifts is None or max_shifts >= K:
        return list(range(K))
    if max_shifts <= 0:
        raise ValueError("max_shifts must be positive or None")

    rng = np.random.default_rng(seed)
    chosen = [0]
    n_extra = min(max_shifts, K) - 1
    if n_extra > 0:
        chosen.extend(sorted(rng.choice([s for s in range(K) if s != 0], size=n_extra, replace=False).astype(int).tolist()))
    return chosen


def load_cyclic_cache(path, data, shifts):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing full cyclic cache:\n{path}\n"
            "Run the scoring script first, or set MODEL_NAMES/BENCHMARK/N_SAMPLES to match an existing cache."
        )

    npz = np.load(path, allow_pickle=True)

    required = ["raw_cyclic", "correct", "model_names", "option_labels", "shifts"]
    missing = [k for k in required if k not in npz.files]
    if missing:
        raise RuntimeError(f"Cyclic cache missing fields {missing}: {path}")

    Q, K, M, S = len(data.questions), len(data.option_labels), len(MODEL_NAMES), len(shifts)
    raw_cyclic = npz["raw_cyclic"].astype(np.float64)

    if raw_cyclic.shape != (Q, K, M, S):
        raise RuntimeError(f"Cache shape mismatch: got {raw_cyclic.shape}, expected {(Q, K, M, S)}")

    cache_model_names = [str(x) for x in npz["model_names"].tolist()]
    if cache_model_names != MODEL_NAMES:
        raise RuntimeError(
            "Cache model_names differ from configured MODEL_NAMES.\n"
            f"Cache:  {cache_model_names}\n"
            f"Script: {MODEL_NAMES}"
        )

    cache_shifts = list(npz["shifts"].astype(int).tolist())
    if cache_shifts != list(map(int, shifts)):
        raise RuntimeError(f"Cache shifts differ: got {cache_shifts}, expected {shifts}")

    if not np.array_equal(npz["correct"].astype(np.int64), data.correct):
        raise RuntimeError("Cache correct labels differ from loaded dataset selection.")

    print(f"Loaded full cyclic run-cache: {path}")
    return raw_cyclic


def get_cyclic_dists(raw_cyclic):
    return np.exp(logsoftmax(raw_cyclic, axis=1))


def pool_cyclic_per_model(p_cyc_qkms):
    return normalize(np.mean(p_cyc_qkms, axis=3), axis=1)


# =============================================================================
# Vectorized RBO p=0.85
# =============================================================================

def compute_rbo_p85_features_fast(p_cyc_qkms):
    """
    p_cyc_qkms shape: Q x K x M x S
    returns: Q x M RBO stability.

    This is vectorized over Q and M.
    Loops only over shift pairs and depth, which are tiny for MC benchmarks.
    """
    Q, K, M, S = p_cyc_qkms.shape

    # order_qmsk[q, m, s, position] = option index at that rank position.
    order_qmsk = np.argsort(-p_cyc_qkms, axis=1, kind="stable").transpose(0, 2, 3, 1)

    rbo = np.zeros((Q, M), dtype=np.float64)

    pair_count = 0
    for si in range(S):
        for sj in range(si + 1, S):
            pair_score = np.zeros((Q, M), dtype=np.float64)

            for d in range(1, K + 1):
                a_top = order_qmsk[:, :, si, :d]
                b_top = order_qmsk[:, :, sj, :d]

                # overlap count between two top-d sets, vectorized over Q,M.
                overlap = (a_top[:, :, :, None] == b_top[:, :, None, :]).sum(axis=(2, 3)).astype(np.float64) / float(d)
                pair_score += (1.0 - RBO_P) * (RBO_P ** (d - 1)) * overlap

            # Complete rankings contain the same option universe, so final overlap is 1.
            pair_score += RBO_P ** K

            rbo += pair_score
            pair_count += 1

    if pair_count == 0:
        return np.ones((Q, M), dtype=np.float64)

    return rbo / float(pair_count)


# =============================================================================
# Vectorized ordinal aggregation
# =============================================================================

def precompute_rank_structures(p_qkm):
    """
    p_qkm shape: Q x K x M

    returns:
      pred_qm: Q x M top-1 predictions.
      ranks_qkm: Q x K x M rank positions, 1 = best.
      order_qmk: Q x M x K ranked option order.
    """
    Q, K, M = p_qkm.shape

    order_qkm = np.argsort(-p_qkm, axis=1, kind="stable")
    ranks_qkm = np.empty_like(order_qkm, dtype=np.int64)

    rank_values = np.arange(1, K + 1, dtype=np.int64)[None, :, None]
    np.put_along_axis(ranks_qkm, order_qkm, rank_values, axis=1)

    pred_qm = order_qkm[:, 0, :].astype(np.int64)
    order_qmk = order_qkm.transpose(0, 2, 1).astype(np.int64)

    return pred_qm, ranks_qkm, order_qmk


def weighted_vote_counts_fast(pred_sub_qm, weights_qm, K):
    Q, M = pred_sub_qm.shape
    scores = np.zeros((Q, K), dtype=np.float64)

    for k in range(K):
        scores[:, k] = np.sum(weights_qm * (pred_sub_qm == k), axis=1)

    return scores


def borda_scores_fast(ranks_sub_qkm, weights_qm, K):
    contrib = K - ranks_sub_qkm.astype(np.float64)
    return np.sum(contrib * weights_qm[:, None, :], axis=2)


def mrr_scores_fast(ranks_sub_qkm, weights_qm):
    contrib = 1.0 / ranks_sub_qkm.astype(np.float64)
    return np.sum(contrib * weights_qm[:, None, :], axis=2)


def borda_tiebreak_scores_fast(order_sub_qmk, weights_qm, K):
    Q, M, _ = order_sub_qmk.shape
    scores = np.zeros((Q, K), dtype=np.float64)
    pos_scores = (K - 1 - np.arange(K, dtype=np.float64))

    for pos in range(K):
        cand_qm = order_sub_qmk[:, :, pos]
        contrib_qm = weights_qm * pos_scores[pos]
        for k in range(K):
            scores[:, k] += np.sum(contrib_qm * (cand_qm == k), axis=1)

    return scores


def instant_runoff_scores_fast(order_sub_qmk, weights_qm, K):
    """
    Vectorized over questions for small K.
    Keeps deterministic tie-breaking close to the original implementation:
    higher count, then higher Borda tie-break score, then lower candidate id.
    """
    Q, M, _ = order_sub_qmk.shape

    remaining = np.ones((Q, K), dtype=bool)
    out = np.zeros((Q, K), dtype=np.float64)

    total_weight = np.sum(weights_qm, axis=1)
    borda_tb = borda_tiebreak_scores_fast(order_sub_qmk, weights_qm, K)

    unresolved = np.ones(Q, dtype=bool)

    for _round in range(K - 1):
        if not np.any(unresolved):
            break

        # For each model, find its highest-ranked still-remaining candidate.
        rem_at_order = np.take_along_axis(
            remaining[:, None, :],
            order_sub_qmk,
            axis=2,
        )
        first_pos = np.argmax(rem_at_order, axis=2)
        first_cand = np.take_along_axis(order_sub_qmk, first_pos[:, :, None], axis=2)[:, :, 0]

        counts = np.zeros((Q, K), dtype=np.float64)
        q_rep = np.repeat(np.arange(Q), M)
        np.add.at(counts, (q_rep, first_cand.reshape(-1)), weights_qm.reshape(-1))

        counts_masked = np.where(remaining, counts, -np.inf)

        # Majority winner.
        best_score = np.max(counts_masked, axis=1)
        best_candidates = np.where(counts_masked == best_score[:, None], 1.0, 0.0)

        # Deterministic tie-breaking among best candidates.
        best_tie = np.where(best_candidates > 0, borda_tb, -np.inf)
        best_tie_score = np.max(best_tie, axis=1)
        best_candidates = best_candidates * (best_tie == best_tie_score[:, None])

        # Remaining ties resolved by smallest candidate id.
        best_cand = np.argmax(best_candidates, axis=1)

        has_majority = unresolved & (best_score > 0.5 * total_weight)
        out[has_majority, best_cand[has_majority]] = 1.0
        unresolved[has_majority] = False

        if not np.any(unresolved):
            break

        # Eliminate candidate with lowest count, then lower Borda, then higher candidate id.
        elim_counts = np.where(remaining & unresolved[:, None], counts, np.inf)
        min_count = np.min(elim_counts, axis=1)
        elim_candidates = np.where(elim_counts == min_count[:, None], 1.0, 0.0)

        elim_tie = np.where(elim_candidates > 0, borda_tb, np.inf)
        min_borda = np.min(elim_tie, axis=1)
        elim_candidates = elim_candidates * (elim_tie == min_borda[:, None])

        # Original min(..., key=(count, borda, -c)) eliminates larger c if still tied.
        cand_ids = np.arange(K, dtype=np.float64)[None, :]
        elim_score = np.where(elim_candidates > 0, cand_ids, -np.inf)
        elim_cand = np.argmax(elim_score, axis=1)

        active_q = np.where(unresolved)[0]
        remaining[active_q, elim_cand[active_q]] = False

        # If only one candidate remains, select it.
        n_remaining = np.sum(remaining, axis=1)
        last = unresolved & (n_remaining == 1)
        if np.any(last):
            last_cand = np.argmax(remaining[last], axis=1)
            out[np.where(last)[0], last_cand] = 1.0
            unresolved[last] = False

    if np.any(unresolved):
        last_cand = np.argmax(remaining[unresolved], axis=1)
        out[np.where(unresolved)[0], last_cand] = 1.0

    return out


def aggregate_scores_precomputed(pred_qm, ranks_qkm, order_qmk, method, subset_idx, weights_qm=None):
    Q, K, M_full = ranks_qkm.shape
    idx = np.asarray(subset_idx, dtype=np.int64)
    m = len(idx)

    if weights_qm is None:
        weights_qm = np.ones((Q, m), dtype=np.float64)

    pred_sub_qm = pred_qm[:, idx]
    ranks_sub_qkm = ranks_qkm[:, :, idx]

    if method == "hard_majority":
        return weighted_vote_counts_fast(pred_sub_qm, weights_qm, K)

    if method == "borda":
        return borda_scores_fast(ranks_sub_qkm, weights_qm, K)

    if method == "mrr":
        return mrr_scores_fast(ranks_sub_qkm, weights_qm)

    if method == "irv":
        order_sub_qmk = order_qmk[:, idx, :]
        return instant_runoff_scores_fast(order_sub_qmk, weights_qm, K)

    raise ValueError(method)


def aggregate_preds_precomputed(pred_qm, ranks_qkm, order_qmk, method, subset_idx, weights_qm=None):
    scores = aggregate_scores_precomputed(
        pred_qm=pred_qm,
        ranks_qkm=ranks_qkm,
        order_qmk=order_qmk,
        method=method,
        subset_idx=subset_idx,
        weights_qm=weights_qm,
    )
    return np.argmax(scores, axis=1).astype(np.int64), scores


def evaluate_pair(preds, base_preds, correct):
    preds = np.asarray(preds, dtype=np.int64)
    base_preds = np.asarray(base_preds, dtype=np.int64)

    method_correct = preds == correct
    base_correct = base_preds == correct

    wins = int(np.sum(method_correct & (~base_correct)))
    losses = int(np.sum((~method_correct) & base_correct))

    return {
        "acc": float(np.mean(method_correct)),
        "base_acc": float(np.mean(base_correct)),
        "delta": float(np.mean(method_correct) - np.mean(base_correct)),
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "mcnemar_p": mcnemar_exact_p(wins, losses) if COMPUTE_ROW_MCNEMAR else float("nan"),
    }


# =============================================================================
# Ensemble-size and heterogeneity analysis
# =============================================================================

def subset_iterator(M, size):
    all_subsets = list(itertools.combinations(range(M), size))

    if MAX_SUBENSEMBLES_PER_SIZE is None or len(all_subsets) <= MAX_SUBENSEMBLES_PER_SIZE:
        return all_subsets

    rng = np.random.default_rng(SUBENSEMBLE_SAMPLE_SEED + size)
    chosen_idx = rng.choice(len(all_subsets), size=MAX_SUBENSEMBLES_PER_SIZE, replace=False)
    return [all_subsets[int(i)] for i in sorted(chosen_idx)]


def compute_single_model_accuracy(pred_qm, correct):
    return np.mean(pred_qm == correct[:, None], axis=0).astype(np.float64)


def evaluate_subset_fast(pred_qm, ranks_qkm, order_qmk, rbo_qm, single_acc_m, correct, subset):
    idx = np.asarray(subset, dtype=np.int64)
    sub_rbo = rbo_qm[:, idx]
    sub_single_acc = single_acc_m[idx]

    weights = softmax_weights(
        zscore_per_question(sub_rbo),
        temperature=STABILITY_TEMPERATURE,
    )

    single_acc_min = float(np.min(sub_single_acc))
    single_acc_max = float(np.max(sub_single_acc))
    single_acc_mean = float(np.mean(sub_single_acc))
    single_acc_std = float(np.std(sub_single_acc))
    single_acc_range = float(single_acc_max - single_acc_min)

    mean_rbo_per_model = np.mean(sub_rbo, axis=0)
    mean_rbo_min = float(np.min(mean_rbo_per_model))
    mean_rbo_max = float(np.max(mean_rbo_per_model))
    mean_rbo_mean = float(np.mean(mean_rbo_per_model))
    mean_rbo_std = float(np.std(mean_rbo_per_model))
    mean_rbo_range = float(mean_rbo_max - mean_rbo_min)

    rows = []

    for method in ORDINAL_AGG_METHODS:
        base_preds, _ = aggregate_preds_precomputed(
            pred_qm=pred_qm,
            ranks_qkm=ranks_qkm,
            order_qmk=order_qmk,
            method=method,
            subset_idx=idx,
            weights_qm=None,
        )

        weighted_preds, _ = aggregate_preds_precomputed(
            pred_qm=pred_qm,
            ranks_qkm=ranks_qkm,
            order_qmk=order_qmk,
            method=method,
            subset_idx=idx,
            weights_qm=weights,
        )

        eval_row = evaluate_pair(weighted_preds, base_preds, correct)

        rows.append({
            "ensemble_size": int(len(idx)),
            "subset_indices": ",".join(map(str, idx.tolist())),
            "subset_models": ",".join(SHORT_NAMES[int(i)] for i in idx.tolist()),
            "base_method": method,
            "weighted_method": f"softmax_stability_rbo_p85_{method}",
            "signal": "rbo_p85",
            "temperature": float(STABILITY_TEMPERATURE),

            "base_acc": eval_row["base_acc"],
            "weighted_acc": eval_row["acc"],
            "delta": eval_row["delta"],
            "wins": eval_row["wins"],
            "losses": eval_row["losses"],
            "net_wins": eval_row["net_wins"],
            "mcnemar_p": eval_row["mcnemar_p"],

            "single_acc_min": single_acc_min,
            "single_acc_max": single_acc_max,
            "single_acc_mean": single_acc_mean,
            "single_acc_std": single_acc_std,
            "single_acc_range": single_acc_range,

            "mean_rbo_min": mean_rbo_min,
            "mean_rbo_max": mean_rbo_max,
            "mean_rbo_mean": mean_rbo_mean,
            "mean_rbo_std": mean_rbo_std,
            "mean_rbo_range": mean_rbo_range,
        })

    return rows


def summarize_rows(rows):
    summary = []

    sizes = sorted(set(r["ensemble_size"] for r in rows))

    for size in sizes:
        for method in ORDINAL_AGG_METHODS:
            block = [r for r in rows if r["ensemble_size"] == size and r["base_method"] == method]
            if not block:
                continue

            base_acc = np.asarray([r["base_acc"] for r in block], dtype=np.float64)
            weighted_acc = np.asarray([r["weighted_acc"] for r in block], dtype=np.float64)
            delta = np.asarray([r["delta"] for r in block], dtype=np.float64)

            single_acc_range = np.asarray([r["single_acc_range"] for r in block], dtype=np.float64)
            single_acc_std = np.asarray([r["single_acc_std"] for r in block], dtype=np.float64)
            mean_rbo_range = np.asarray([r["mean_rbo_range"] for r in block], dtype=np.float64)
            mean_rbo_std = np.asarray([r["mean_rbo_std"] for r in block], dtype=np.float64)

            best_i = int(np.argmax(delta))
            worst_i = int(np.argmin(delta))

            summary.append({
                "ensemble_size": int(size),
                "base_method": method,
                "n_subensembles": int(len(block)),

                "base_acc_mean": float(np.mean(base_acc)),
                "base_acc_std": float(np.std(base_acc)),
                "base_acc_q05": float(np.quantile(base_acc, 0.05)),
                "base_acc_q25": float(np.quantile(base_acc, 0.25)),
                "base_acc_q50": float(np.quantile(base_acc, 0.50)),
                "base_acc_q75": float(np.quantile(base_acc, 0.75)),
                "base_acc_q95": float(np.quantile(base_acc, 0.95)),

                "weighted_acc_mean": float(np.mean(weighted_acc)),
                "weighted_acc_std": float(np.std(weighted_acc)),
                "weighted_acc_q05": float(np.quantile(weighted_acc, 0.05)),
                "weighted_acc_q25": float(np.quantile(weighted_acc, 0.25)),
                "weighted_acc_q50": float(np.quantile(weighted_acc, 0.50)),
                "weighted_acc_q75": float(np.quantile(weighted_acc, 0.75)),
                "weighted_acc_q95": float(np.quantile(weighted_acc, 0.95)),

                "delta_mean": float(np.mean(delta)),
                "delta_std": float(np.std(delta)),
                "delta_q05": float(np.quantile(delta, 0.05)),
                "delta_q25": float(np.quantile(delta, 0.25)),
                "delta_q50": float(np.quantile(delta, 0.50)),
                "delta_q75": float(np.quantile(delta, 0.75)),
                "delta_q95": float(np.quantile(delta, 0.95)),

                "frac_delta_positive": float(np.mean(delta > 0.0)),
                "frac_delta_nonnegative": float(np.mean(delta >= 0.0)),

                "single_acc_range_mean": float(np.mean(single_acc_range)),
                "single_acc_std_mean": float(np.mean(single_acc_std)),
                "mean_rbo_range_mean": float(np.mean(mean_rbo_range)),
                "mean_rbo_std_mean": float(np.mean(mean_rbo_std)),

                "corr_single_acc_range_delta": safe_corr(single_acc_range, delta),
                "corr_single_acc_std_delta": safe_corr(single_acc_std, delta),
                "corr_mean_rbo_range_delta": safe_corr(mean_rbo_range, delta),
                "corr_mean_rbo_std_delta": safe_corr(mean_rbo_std, delta),

                "best_delta": float(delta[best_i]),
                "best_subset_models": block[best_i]["subset_models"],
                "worst_delta": float(delta[worst_i]),
                "worst_subset_models": block[worst_i]["subset_models"],
            })

    return summary


def best_rows_by_size(summary_rows):
    out = []
    for size in sorted(set(r["ensemble_size"] for r in summary_rows)):
        block = [r for r in summary_rows if r["ensemble_size"] == size]
        out.append(max(block, key=lambda r: r["weighted_acc_mean"]))
    return out


def heterogeneity_correlation_rows(rows):
    out = []

    metrics = [
        "single_acc_std",
        "single_acc_range",
        "single_acc_min",
        "single_acc_max",
        "mean_rbo_std",
        "mean_rbo_range",
        "mean_rbo_mean",
    ]

    sizes = sorted(set(r["ensemble_size"] for r in rows))

    for size in sizes:
        for method in ORDINAL_AGG_METHODS:
            block = [
                r for r in rows
                if r["ensemble_size"] == size and r["base_method"] == method
            ]

            if len(block) < 2:
                continue

            delta = np.asarray([r["delta"] for r in block], dtype=np.float64)
            weighted_acc = np.asarray([r["weighted_acc"] for r in block], dtype=np.float64)
            base_acc = np.asarray([r["base_acc"] for r in block], dtype=np.float64)

            for metric in metrics:
                x = np.asarray([r[metric] for r in block], dtype=np.float64)

                out.append({
                    "ensemble_size": int(size),
                    "base_method": method,
                    "heterogeneity_metric": metric,
                    "n_subensembles": int(len(block)),
                    "corr_metric_delta": safe_corr(x, delta),
                    "corr_metric_weighted_acc": safe_corr(x, weighted_acc),
                    "corr_metric_base_acc": safe_corr(x, base_acc),
                    "metric_mean": float(np.mean(x)),
                    "metric_std": float(np.std(x)),
                    "delta_mean": float(np.mean(delta)),
                    "weighted_acc_mean": float(np.mean(weighted_acc)),
                    "base_acc_mean": float(np.mean(base_acc)),
                })

    return out


def assign_quantile_bins(x, n_bins):
    x = np.asarray(x, dtype=np.float64)
    qs = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    return np.searchsorted(qs[1:-1], x, side="right").astype(np.int64)


def heterogeneity_bin_rows(rows, metric="single_acc_range", n_bins=3):
    out = []

    sizes = sorted(set(r["ensemble_size"] for r in rows))

    for size in sizes:
        for method in ORDINAL_AGG_METHODS:
            block = [
                r for r in rows
                if r["ensemble_size"] == size and r["base_method"] == method
            ]

            if len(block) < n_bins:
                continue

            x = np.asarray([r[metric] for r in block], dtype=np.float64)
            bins = assign_quantile_bins(x, n_bins=n_bins)

            for b in range(n_bins):
                bb = [r for r, bi in zip(block, bins) if int(bi) == b]
                if not bb:
                    continue

                base_acc = np.asarray([r["base_acc"] for r in bb], dtype=np.float64)
                weighted_acc = np.asarray([r["weighted_acc"] for r in bb], dtype=np.float64)
                delta = np.asarray([r["delta"] for r in bb], dtype=np.float64)
                metric_vals = np.asarray([r[metric] for r in bb], dtype=np.float64)

                out.append({
                    "ensemble_size": int(size),
                    "base_method": method,
                    "heterogeneity_metric": metric,
                    "heterogeneity_bin": int(b),
                    "heterogeneity_bin_name": ["low", "medium", "high"][b] if n_bins == 3 else f"bin_{b}",
                    "n_subensembles": int(len(bb)),
                    "metric_min": float(np.min(metric_vals)),
                    "metric_max": float(np.max(metric_vals)),
                    "metric_mean": float(np.mean(metric_vals)),
                    "base_acc_mean": float(np.mean(base_acc)),
                    "weighted_acc_mean": float(np.mean(weighted_acc)),
                    "delta_mean": float(np.mean(delta)),
                    "delta_q25": float(np.quantile(delta, 0.25)),
                    "delta_q50": float(np.quantile(delta, 0.50)),
                    "delta_q75": float(np.quantile(delta, 0.75)),
                    "frac_delta_positive": float(np.mean(delta > 0.0)),
                })

    return out


# =============================================================================
# Main
# =============================================================================

def main():
    data = load_dataset_pack(N_SAMPLES, BENCHMARK, DATASET_SEED)
    correct = data.correct

    K = len(data.option_labels)
    cyclic_shifts = choose_cyclic_shifts(K=K, max_shifts=MAX_CYCLIC_SHIFTS, seed=CYCLIC_SHIFT_SEED)
    if 0 not in cyclic_shifts:
        raise RuntimeError("cyclic_shifts must include 0.")

    raw_cyclic = load_cyclic_cache(CYCLIC_NPZ_FILE, data, cyclic_shifts)

    p_cyc_qkms = get_cyclic_dists(raw_cyclic)
    p_cycmean_qkm = pool_cyclic_per_model(p_cyc_qkms)

    Q, K, M = p_cycmean_qkm.shape

    print("\n" + "=" * 120)
    print("Ensemble-size + heterogeneity ablation from cached cyclic scores")
    print("=" * 120)
    print(f"N={Q} | K={K} | M={M} | benchmark={data.benchmark_resolved} | seed={DATASET_SEED}")
    print(f"Cache tag: {CACHE_TAG}")
    print(f"Cyclic cache: {CYCLIC_NPZ_FILE}")
    print(f"Ordinal methods: {ORDINAL_AGG_METHODS}")
    print(f"Stability signal: RBO p={RBO_P}")
    print(f"Temperature: T={STABILITY_TEMPERATURE}")
    print(f"All sub-ensembles: {'yes' if MAX_SUBENSEMBLES_PER_SIZE is None else 'sampled'}")
    print(f"Row-level McNemar: {COMPUTE_ROW_MCNEMAR}")

    print("\nPrecomputing cyclic-mean rank structures ...")
    pred_qm, ranks_qkm, order_qmk = precompute_rank_structures(p_cycmean_qkm)
    single_acc_m = compute_single_model_accuracy(pred_qm, correct)

    print("\nSingle-model cyclic-mean accuracies used for heterogeneity:")
    for mi, acc in enumerate(single_acc_m):
        print(f"  {SHORT_NAMES[mi]}: {acc:.4f}")

    print("\nComputing vectorized RBO p=0.85 stability features ...")
    rbo_qm = compute_rbo_p85_features_fast(p_cyc_qkms)

    print("\nComputing signal validity analysis at model-question level ...")
    signal_validity_summary, signal_validity_model_rows, signal_validity_mq_rows = compute_signal_validity_rows(
        rbo_qm=rbo_qm,
        pred_qm=pred_qm,
        correct=correct,
        short_names=SHORT_NAMES,
    )

    print_table(
        [signal_validity_summary],
        "Signal validity summary",
        headers=[
            "n_questions",
            "n_models",
            "n_model_question_predictions",
            "phi_correct_mean",
            "phi_wrong_mean",
            "phi_correct_minus_wrong",
            "point_biserial_r",
            "point_biserial_p",
            "within_model_r_min",
            "within_model_r_max",
            "pooled_within_model_centered_r",
        ],
        nd=6,
    )

    print_table(
        signal_validity_model_rows,
        "Signal validity by model",
        headers=[
            "model_index",
            "model",
            "accuracy",
            "phi_mean",
            "phi_correct_mean",
            "phi_wrong_mean",
            "phi_correct_minus_wrong",
            "within_model_r",
            "within_model_p",
        ],
        nd=6,
        max_rows=None,
    )

    rows = []
    for size in range(1, M + 1):
        subsets = subset_iterator(M, size)
        print(f"Evaluating ensemble size {size}/{M}: {len(subsets)} subsets")

        for subset in subsets:
            rows.extend(evaluate_subset_fast(
                pred_qm=pred_qm,
                ranks_qkm=ranks_qkm,
                order_qmk=order_qmk,
                rbo_qm=rbo_qm,
                single_acc_m=single_acc_m,
                correct=correct,
                subset=subset,
            ))

    summary_rows = summarize_rows(rows)
    best_summary_rows = best_rows_by_size(summary_rows)

    heterogeneity_corr_rows = heterogeneity_correlation_rows(rows)
    heterogeneity_bin_single_range_rows = heterogeneity_bin_rows(
        rows,
        metric="single_acc_range",
        n_bins=HETEROGENEITY_N_BINS,
    )
    heterogeneity_bin_single_std_rows = heterogeneity_bin_rows(
        rows,
        metric="single_acc_std",
        n_bins=HETEROGENEITY_N_BINS,
    )

    print_table(
        best_summary_rows,
        "Best mean weighted accuracy per ensemble size",
        headers=[
            "ensemble_size", "base_method", "n_subensembles",
            "base_acc_mean", "weighted_acc_mean", "delta_mean",
            "delta_q05", "delta_q50", "delta_q95",
            "frac_delta_positive",
            "corr_single_acc_range_delta",
        ],
        max_rows=None,
    )

    for method in ["hard_majority", "borda", "mrr", "irv"]:
        print_table(
            [r for r in summary_rows if r["base_method"] == method],
            f"{method} ensemble-size ablation",
            headers=[
                "ensemble_size", "n_subensembles",
                "base_acc_mean", "weighted_acc_mean", "delta_mean",
                "delta_q05", "delta_q50", "delta_q95",
                "frac_delta_positive",
                "single_acc_range_mean",
                "corr_single_acc_range_delta",
            ],
            max_rows=None,
        )

    print_table(
        [
            r for r in heterogeneity_corr_rows
            if r["base_method"] == HETEROGENEITY_PRIMARY_METHOD
            and r["heterogeneity_metric"] in ["single_acc_range", "single_acc_std", "mean_rbo_range", "mean_rbo_std"]
        ],
        f"Heterogeneity correlations ({HETEROGENEITY_PRIMARY_METHOD})",
        headers=[
            "ensemble_size", "base_method", "heterogeneity_metric", "n_subensembles",
            "corr_metric_delta", "corr_metric_weighted_acc", "corr_metric_base_acc",
            "metric_mean", "metric_std", "delta_mean",
        ],
        max_rows=None,
    )

    print_table(
        [
            r for r in heterogeneity_bin_single_range_rows
            if r["base_method"] == HETEROGENEITY_PRIMARY_METHOD
        ],
        f"Heterogeneity bins by single-model accuracy range ({HETEROGENEITY_PRIMARY_METHOD})",
        headers=[
            "ensemble_size", "base_method", "heterogeneity_bin_name",
            "n_subensembles", "metric_mean",
            "base_acc_mean", "weighted_acc_mean", "delta_mean",
            "delta_q25", "delta_q50", "delta_q75", "frac_delta_positive",
        ],
        max_rows=None,
    )

    row_headers = [
        "ensemble_size", "subset_indices", "subset_models",
        "base_method", "weighted_method", "signal", "temperature",
        "base_acc", "weighted_acc", "delta",
        "wins", "losses", "net_wins", "mcnemar_p",
        "single_acc_min", "single_acc_max", "single_acc_mean", "single_acc_std", "single_acc_range",
        "mean_rbo_min", "mean_rbo_max", "mean_rbo_mean", "mean_rbo_std", "mean_rbo_range",
    ]

    summary_headers = [
        "ensemble_size", "base_method", "n_subensembles",
        "base_acc_mean", "base_acc_std", "base_acc_q05", "base_acc_q25", "base_acc_q50", "base_acc_q75", "base_acc_q95",
        "weighted_acc_mean", "weighted_acc_std", "weighted_acc_q05", "weighted_acc_q25", "weighted_acc_q50", "weighted_acc_q75", "weighted_acc_q95",
        "delta_mean", "delta_std", "delta_q05", "delta_q25", "delta_q50", "delta_q75", "delta_q95",
        "frac_delta_positive", "frac_delta_nonnegative",
        "single_acc_range_mean", "single_acc_std_mean", "mean_rbo_range_mean", "mean_rbo_std_mean",
        "corr_single_acc_range_delta", "corr_single_acc_std_delta", "corr_mean_rbo_range_delta", "corr_mean_rbo_std_delta",
        "best_delta", "best_subset_models",
        "worst_delta", "worst_subset_models",
    ]

    heterogeneity_corr_headers = [
        "ensemble_size", "base_method", "heterogeneity_metric", "n_subensembles",
        "corr_metric_delta", "corr_metric_weighted_acc", "corr_metric_base_acc",
        "metric_mean", "metric_std", "delta_mean", "weighted_acc_mean", "base_acc_mean",
    ]

    heterogeneity_bin_headers = [
        "ensemble_size", "base_method", "heterogeneity_metric",
        "heterogeneity_bin", "heterogeneity_bin_name", "n_subensembles",
        "metric_min", "metric_max", "metric_mean",
        "base_acc_mean", "weighted_acc_mean", "delta_mean",
        "delta_q25", "delta_q50", "delta_q75", "frac_delta_positive",
    ]

    signal_validity_summary_headers = [
        "n_questions",
        "n_models",
        "n_model_question_predictions",
        "phi_correct_mean",
        "phi_wrong_mean",
        "phi_correct_minus_wrong",
        "point_biserial_r",
        "point_biserial_p",
        "within_model_r_min",
        "within_model_r_max",
        "pooled_within_model_centered_r",
    ]

    signal_validity_model_headers = [
        "model_index",
        "model",
        "n_questions",
        "accuracy",
        "phi_mean",
        "phi_correct_mean",
        "phi_wrong_mean",
        "phi_correct_minus_wrong",
        "within_model_r",
        "within_model_p",
    ]

    signal_validity_mq_headers = [
        "question_index",
        "model_index",
        "model",
        "phi",
        "correct",
        "prediction",
        "gold",
    ]

    write_csv(
        SIGNAL_VALIDITY_CSV_FILE,
        [signal_validity_summary],
        headers=signal_validity_summary_headers,
    )

    write_csv(
        SIGNAL_VALIDITY_MODEL_CSV_FILE,
        signal_validity_model_rows,
        headers=signal_validity_model_headers,
    )

    # Optional but useful for debugging / audit.
    # This file has Q*M rows.
    write_csv(
        os.path.join(OUT_DIR, f"signal_validity_model_question_rows_{CACHE_TAG}.csv"),
        signal_validity_mq_rows,
        headers=signal_validity_mq_headers,
    )

    write_csv(ROWS_CSV_FILE, rows, headers=row_headers)
    write_csv(SUMMARY_CSV_FILE, summary_rows, headers=summary_headers)
    write_csv(HETEROGENEITY_CORR_CSV_FILE, heterogeneity_corr_rows, headers=heterogeneity_corr_headers)
    write_csv(HETEROGENEITY_BIN_RANGE_CSV_FILE, heterogeneity_bin_single_range_rows, headers=heterogeneity_bin_headers)
    write_csv(HETEROGENEITY_BIN_STD_CSV_FILE, heterogeneity_bin_single_std_rows, headers=heterogeneity_bin_headers)

    results = {
        "config": {
            "model_names": MODEL_NAMES,
            "short_names": SHORT_NAMES,
            "benchmark": BENCHMARK,
            "benchmark_resolved": data.benchmark_resolved,
            "N": Q,
            "K": K,
            "M": M,
            "dataset_seed": DATASET_SEED,
            "cache_tag": CACHE_TAG,
            "cyclic_npz_file": CYCLIC_NPZ_FILE,
            "cyclic_shifts": cyclic_shifts,
            "ordinal_methods": ORDINAL_AGG_METHODS,
            "stability_signal": "rbo_p85",
            "rbo_p": RBO_P,
            "stability_temperature": STABILITY_TEMPERATURE,
            "max_subensembles_per_size": MAX_SUBENSEMBLES_PER_SIZE,
            "compute_row_mcnemar": COMPUTE_ROW_MCNEMAR,
            "heterogeneity_n_bins": HETEROGENEITY_N_BINS,
            "heterogeneity_primary_method": HETEROGENEITY_PRIMARY_METHOD,
        },
        "single_model_cyclic_mean_acc": {
            SHORT_NAMES[i]: float(single_acc_m[i])
            for i in range(M)
        },
        "summary_rows": summary_rows,
        "best_summary_rows": best_summary_rows,
        "heterogeneity_correlation_rows": heterogeneity_corr_rows,
        "heterogeneity_bin_single_acc_range_rows": heterogeneity_bin_single_range_rows,
        "heterogeneity_bin_single_acc_std_rows": heterogeneity_bin_single_std_rows,
        "row_csv": ROWS_CSV_FILE,
        "summary_csv": SUMMARY_CSV_FILE,
        "heterogeneity_corr_csv": HETEROGENEITY_CORR_CSV_FILE,
        "heterogeneity_bin_range_csv": HETEROGENEITY_BIN_RANGE_CSV_FILE,
        "heterogeneity_bin_std_csv": HETEROGENEITY_BIN_STD_CSV_FILE,
        "signal_validity_summary": signal_validity_summary,
        "signal_validity_by_model": signal_validity_model_rows,
        "signal_validity_csv": SIGNAL_VALIDITY_CSV_FILE,
        "signal_validity_model_csv": SIGNAL_VALIDITY_MODEL_CSV_FILE,
    }

    with open(JSON_FILE, "w") as f:
        json.dump(serialise(results), f, indent=2)

    print(f"\nSaved row-level CSV:             {ROWS_CSV_FILE}")
    print(f"Saved summary CSV:               {SUMMARY_CSV_FILE}")
    print(f"Saved heterogeneity corr CSV:    {HETEROGENEITY_CORR_CSV_FILE}")
    print(f"Saved heterogeneity range CSV:   {HETEROGENEITY_BIN_RANGE_CSV_FILE}")
    print(f"Saved heterogeneity std CSV:     {HETEROGENEITY_BIN_STD_CSV_FILE}")
    print(f"Saved JSON:                      {JSON_FILE}")
    print(f"Saved signal validity summary:    {SIGNAL_VALIDITY_CSV_FILE}")
    print(f"Saved signal validity by model:   {SIGNAL_VALIDITY_MODEL_CSV_FILE}")


if __name__ == "__main__":
    main()