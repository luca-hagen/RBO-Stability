import json
import math
import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
from datasets import load_dataset


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAMES = [
    #"Qwen/Qwen3-0.6B",
    #"Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-270m-it",
    #"meta-llama/Llama-3.2-1B-Instruct",
    #"meta-llama/Llama-3.2-3B-Instruct",
    #"google/gemma-3-1b-it",
    #"HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
SHORT_NAMES = [m.split("/")[-1] for m in MODEL_NAMES]

BENCHMARK = "mmlu"       # "mmlu_pro" or "mmlu"
N_SAMPLES = 14042        # MMLU-Pro: 9981; MMLU: 14042
DATASET_SEED = 42
DATASET_K_FILTER: Union[str, int] = "modal"

OUT_DIR = "./majority_regime_analysis_results"
CACHE_TAG = f"{BENCHMARK}_K{DATASET_K_FILTER}_N{N_SAMPLES}_seed{DATASET_SEED}_M{len(MODEL_NAMES)}"

MAX_CYCLIC_SHIFTS = None
CYCLIC_SHIFT_SEED = 123
CYCLIC_TAG = (
    f"{CACHE_TAG}_cyclic"
    f"_S{'all' if MAX_CYCLIC_SHIFTS is None else MAX_CYCLIC_SHIFTS}"
    f"_shiftseed{CYCLIC_SHIFT_SEED}"
)
CYCLIC_NPZ_FILE = os.path.join(OUT_DIR, f"cyclic_gen_scores_{CYCLIC_TAG}.npz")

EPS = 1e-6
RBO_P = 0.85
STABILITY_TEMPERATURE = 1.0
ENTROPY_TEMPERATURE = 1.0

AGG_METHODS = [
    "hard_majority",
    "borda",
    "mrr",
    "irv",
    "arithmetic_mean",
    "geometric_mean",
]
ORDINAL_AGG_METHODS = ["hard_majority", "borda", "mrr", "irv"]

# Toy experiment:
# tau < 1 sharpens probabilities and simulates an increasingly overconfident
# entropy signal for the weakest model.
#
# Crucially, this script uses the sharpened probabilities ONLY to compute entropy
# weights. The actual aggregators still aggregate the unchanged cyclic-mean
# probabilities. This isolates calibration sensitivity of entropy weighting.
TOY_ENTROPY_SIGNAL_TAU_GRID = [1.0, 0.75, 0.5, 0.333333, 0.25, 0.2, 0.166667, 0.125, 0.1, 0.05, 0.02, 0.01]


# =============================================================================
# Dataset loading
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


def canonical_benchmark_name(benchmark: str) -> str:
    b = benchmark.strip().lower().replace("-", "_")
    aliases = {
        "mmlupro": "mmlu_pro",
        "mmlu_pro": "mmlu_pro",
        "mmlu": "mmlu",
    }
    if b not in aliases:
        raise ValueError(f"This compact script currently supports only MMLU/MMLU-Pro, got {benchmark!r}")
    return aliases[b]


def choose_target_k(option_lengths, benchmark):
    if isinstance(DATASET_K_FILTER, int):
        return int(DATASET_K_FILTER), f"explicit_{int(DATASET_K_FILTER)}"

    mode = str(DATASET_K_FILTER).lower()

    if mode == "modal":
        vals, counts = np.unique(np.asarray(option_lengths, dtype=np.int64), return_counts=True)
        return int(vals[np.argmax(counts)]), "modal"

    if mode == "max":
        return int(max(option_lengths)), "max"

    if mode == "none":
        unique = sorted(set(int(x) for x in option_lengths))
        if len(unique) != 1:
            raise ValueError(f"DATASET_K_FILTER='none' requires uniform K, got {unique}")
        return int(unique[0]), "none_uniform"

    if benchmark == "mmlu":
        return 4, "benchmark_fixed_4"

    raise ValueError(f"Unknown DATASET_K_FILTER={DATASET_K_FILTER!r}")


def remap_to_standard_labels(options, correct_idx):
    if not (0 <= int(correct_idx) < len(options)):
        raise ValueError(f"correct_idx out of range: {correct_idx}")
    return [str(o) for o in options], int(correct_idx)


def load_dataset_pack(n_samples, benchmark, seed):
    benchmark = canonical_benchmark_name(benchmark)
    examples = []

    if benchmark == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        for i, ex in enumerate(ds):
            examples.append(
                (i, str(ex["question"]), [str(o) for o in ex["options"]], int(ex["answer_index"]))
            )

    elif benchmark == "mmlu":
        ds = load_dataset("cais/mmlu", name="all", split="test")
        for i, ex in enumerate(ds):
            examples.append(
                (i, str(ex["question"]), [str(o) for o in ex["choices"]], int(ex["answer"]))
            )

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

        qid = stable_json_hash({
            "benchmark": benchmark,
            "source_index": int(source_i),
            "question": q,
            "options": opts,
            "correct": int(a),
            "k": int(target_k),
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
# Numeric helpers
# =============================================================================

def logsoftmax(x, axis):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    z = np.exp(x - m)
    z /= np.maximum(np.sum(z, axis=axis, keepdims=True), EPS)
    return np.log(np.maximum(z, EPS))


def normalize(p, axis=None):
    p = np.maximum(np.asarray(p, dtype=np.float64), EPS)
    return p / np.maximum(np.sum(p, axis=axis, keepdims=True), EPS)


def entropy(p, axis=-1):
    p = normalize(p, axis=axis)
    return -np.sum(p * np.log(np.maximum(p, EPS)), axis=axis)


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


def rank_positions(scores, axis=-1):
    x = np.asarray(scores)
    axis = axis % x.ndim

    order = np.argsort(-x, axis=axis, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)

    idx = np.indices(order.shape, sparse=True)
    idx = list(idx)
    idx[axis] = order

    rank_values = np.arange(1, x.shape[axis] + 1, dtype=np.int64)
    shape = [1] * x.ndim
    shape[axis] = x.shape[axis]

    ranks[tuple(idx)] = rank_values.reshape(shape)
    return ranks


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
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        - n * math.log(2.0)
        for i in range(k + 1)
    ]
    m = max(log_terms)
    cdf = math.exp(m) * sum(math.exp(t - m) for t in log_terms)
    return float(min(1.0, 2.0 * cdf))


def fmt_float(x, nd=4):
    try:
        if np.isnan(x):
            return "nan"
    except Exception:
        pass
    return f"{float(x):.{nd}f}"


def print_table(rows, title, headers=None, nd=4):
    print(f"\n--- {title} ---")
    if not rows:
        print("<empty>")
        return

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


def compare_two_prediction_sets(preds_a, preds_b, correct):
    preds_a = np.asarray(preds_a, dtype=np.int64)
    preds_b = np.asarray(preds_b, dtype=np.int64)
    correct = np.asarray(correct, dtype=np.int64)

    a_correct = preds_a == correct
    b_correct = preds_b == correct

    wins_a = int(np.sum(a_correct & (~b_correct)))
    wins_b = int(np.sum(b_correct & (~a_correct)))

    return {
        "acc_a": float(np.mean(a_correct)),
        "acc_b": float(np.mean(b_correct)),
        "delta_a_minus_b": float(np.mean(a_correct) - np.mean(b_correct)),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "mcnemar_p": mcnemar_exact_p(wins_a, wins_b),
    }


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
        chosen.extend(sorted(
            rng.choice([s for s in range(K) if s != 0], size=n_extra, replace=False)
            .astype(int)
            .tolist()
        ))
    return chosen


def load_cyclic_cache(path, data, shifts):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cyclic cache:\n{path}\n"
            "Run the scoring script first, or set BENCHMARK/N_SAMPLES/DATASET_K_FILTER to match an existing cache."
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

    print(f"Loaded cyclic cache: {path}")
    return raw_cyclic


def get_cyclic_dists(raw_cyclic):
    return np.exp(logsoftmax(raw_cyclic, axis=1))


def pool_cyclic_per_model(p_cyc_qkms):
    return normalize(np.mean(p_cyc_qkms, axis=3), axis=1)


# =============================================================================
# RBO stability
# =============================================================================

def compute_rbo_features(p_cyc_qkms, p=0.85):
    """
    p_cyc_qkms shape: Q x K x M x S
    returns Q x M RBO stability.
    """
    Q, K, M, S = p_cyc_qkms.shape
    order_qmsk = np.argsort(-p_cyc_qkms, axis=1, kind="stable").transpose(0, 2, 3, 1)

    rbo = np.zeros((Q, M), dtype=np.float64)
    pair_count = 0

    for si in range(S):
        for sj in range(si + 1, S):
            pair_score = np.zeros((Q, M), dtype=np.float64)

            for d in range(1, K + 1):
                a_top = order_qmsk[:, :, si, :d]
                b_top = order_qmsk[:, :, sj, :d]
                overlap = (
                    a_top[:, :, :, None] == b_top[:, :, None, :]
                ).sum(axis=(2, 3)).astype(np.float64) / float(d)

                pair_score += (1.0 - p) * (p ** (d - 1)) * overlap

            pair_score += p ** K
            rbo += pair_score
            pair_count += 1

    if pair_count == 0:
        return np.ones((Q, M), dtype=np.float64)

    return rbo / float(pair_count)


# =============================================================================
# Aggregation
# =============================================================================

def weighted_vote_counts(pred_qm, weights_qm, K):
    Q, M = pred_qm.shape
    scores = np.zeros((Q, K), dtype=np.float64)

    q_idx = np.repeat(np.arange(Q), M)
    k_idx = pred_qm.reshape(-1)
    w = weights_qm.reshape(-1)

    np.add.at(scores, (q_idx, k_idx), w)
    return scores


def rank_order_per_model(p_qkm):
    return np.argsort(-p_qkm, axis=1, kind="stable").transpose(0, 2, 1)


def borda_tiebreak_scores_from_rankings(order_qmk, weights_qm, K):
    Q, M, _ = order_qmk.shape
    scores = np.zeros((Q, K), dtype=np.float64)
    pos_scores = K - 1 - np.arange(K, dtype=np.float64)

    contrib = weights_qm[:, :, None] * pos_scores[None, None, :]

    q_idx = np.repeat(np.arange(Q), M * K)
    cand_idx = order_qmk.reshape(-1)
    val = contrib.reshape(-1)

    np.add.at(scores, (q_idx, cand_idx), val)
    return scores


def instant_runoff_scores(p_qkm, weights_qm=None):
    Q, K, M = p_qkm.shape
    if weights_qm is None:
        weights_qm = np.ones((Q, M), dtype=np.float64)

    order_qmk = rank_order_per_model(p_qkm)
    borda_tb = borda_tiebreak_scores_from_rankings(order_qmk, weights_qm, K)
    out = np.zeros((Q, K), dtype=np.float64)

    for qi in range(Q):
        remaining = set(range(K))
        total_weight = float(np.sum(weights_qm[qi]))

        while len(remaining) > 1:
            counts = {cand: 0.0 for cand in remaining}

            for mi in range(M):
                w = float(weights_qm[qi, mi])
                for cand in order_qmk[qi, mi]:
                    cand = int(cand)
                    if cand in remaining:
                        counts[cand] += w
                        break

            best_cand = max(remaining, key=lambda c: (counts[c], borda_tb[qi, c], -c))
            if counts[best_cand] > 0.5 * total_weight:
                out[qi, best_cand] = 1.0
                break

            elim_cand = min(remaining, key=lambda c: (counts[c], borda_tb[qi, c], -c))
            remaining.remove(elim_cand)

        if np.sum(out[qi]) == 0.0:
            out[qi, next(iter(remaining))] = 1.0

    return out


def aggregate_scores(p_qkm, method, weights_qm=None):
    Q, K, M = p_qkm.shape

    if weights_qm is None:
        weights_qm = np.ones((Q, M), dtype=np.float64)
    else:
        weights_qm = np.asarray(weights_qm, dtype=np.float64)

    if method == "hard_majority":
        pred_qm = np.argmax(p_qkm, axis=1)
        return weighted_vote_counts(pred_qm, weights_qm, K)

    if method == "borda":
        ranks_qkm = rank_positions(p_qkm, axis=1).astype(np.float64)
        contrib = K - ranks_qkm
        return np.sum(contrib * weights_qm[:, None, :], axis=2)

    if method == "mrr":
        ranks_qkm = rank_positions(p_qkm, axis=1).astype(np.float64)
        contrib = 1.0 / ranks_qkm
        return np.sum(contrib * weights_qm[:, None, :], axis=2)

    if method == "irv":
        return instant_runoff_scores(p_qkm, weights_qm=weights_qm)

    if method == "arithmetic_mean":
        denom = np.maximum(np.sum(weights_qm, axis=1, keepdims=True), EPS)
        return np.sum(p_qkm * weights_qm[:, None, :], axis=2) / denom

    if method == "geometric_mean":
        denom = np.maximum(np.sum(weights_qm, axis=1, keepdims=True), EPS)
        logp = np.log(np.maximum(p_qkm, EPS))
        return np.sum(logp * weights_qm[:, None, :], axis=2) / denom

    raise ValueError(method)


def aggregate_preds(p_qkm, method, weights_qm=None):
    scores = aggregate_scores(p_qkm, method, weights_qm=weights_qm)
    preds = np.argmax(scores, axis=1).astype(np.int64)
    return preds, scores


def make_entropy_weights(p_qkm_for_entropy_signal, temperature=1.0):
    H_qm = entropy(np.transpose(p_qkm_for_entropy_signal, (0, 2, 1)), axis=2)
    confidence_qm = -H_qm
    return softmax_weights(zscore_per_question(confidence_qm), temperature=temperature)


def make_rbo_weights(rbo_qm, temperature=1.0):
    return softmax_weights(zscore_per_question(rbo_qm), temperature=temperature)


# =============================================================================
# Toy helpers
# =============================================================================

def sharpen_probs_by_temperature(p_qk, tau):
    """
    Temperature-sharpen a probability vector without changing its ranking.

    tau < 1: sharper / more overconfident
    tau = 1: unchanged
    tau > 1: flatter

    Equivalent to applying softmax(log(p) / tau).
    """
    p_qk = normalize(np.asarray(p_qk, dtype=np.float64), axis=1)
    logp = np.log(np.maximum(p_qk, EPS))
    return np.exp(logsoftmax(logp / max(float(tau), EPS), axis=1))


def compute_single_model_acc(p_qkm, correct):
    pred_qm = np.argmax(p_qkm, axis=1)
    return np.mean(pred_qm == correct[:, None], axis=0)


def top1_signature(p_qkm):
    return np.argmax(np.asarray(p_qkm, dtype=np.float64), axis=1)


def rank_signature(p_qkm):
    return np.argsort(-np.asarray(p_qkm, dtype=np.float64), axis=1, kind="stable").transpose(0, 2, 1)


# =============================================================================
# Analyses
# =============================================================================

def paired_entropy_vs_rbo_table(p_qkm, rbo_qm, correct, title):
    ent_w = make_entropy_weights(p_qkm, temperature=ENTROPY_TEMPERATURE)
    rbo_w = make_rbo_weights(rbo_qm, temperature=STABILITY_TEMPERATURE)

    rows = []

    for method in AGG_METHODS:
        base_preds, _ = aggregate_preds(p_qkm, method, weights_qm=None)
        ent_preds, _ = aggregate_preds(p_qkm, method, weights_qm=ent_w)
        rbo_preds, _ = aggregate_preds(p_qkm, method, weights_qm=rbo_w)

        base_correct = base_preds == correct
        ent_correct = ent_preds == correct
        rbo_correct = rbo_preds == correct

        pair = compare_two_prediction_sets(ent_preds, rbo_preds, correct)

        base_acc = float(np.mean(base_correct))
        ent_acc = float(np.mean(ent_correct))
        rbo_acc = float(np.mean(rbo_correct))

        rows.append({
            "method": method,
            "base_acc": base_acc,
            "entropy_acc": ent_acc,
            "rbo_acc": rbo_acc,
            "entropy_delta": ent_acc - base_acc,
            "rbo_delta": rbo_acc - base_acc,
            "entropy_minus_rbo": ent_acc - rbo_acc,
            "entropy_wins": pair["wins_a"],
            "rbo_wins": pair["wins_b"],
            "mcnemar_p_entropy_vs_rbo": pair["mcnemar_p"],
        })

    print_table(
        rows,
        title,
        headers=[
            "method",
            "base_acc",
            "entropy_acc",
            "rbo_acc",
            "entropy_delta",
            "rbo_delta",
            "entropy_minus_rbo",
            "entropy_wins",
            "rbo_wins",
            "mcnemar_p_entropy_vs_rbo",
        ],
        nd=6,
    )
    return rows


def toy_overconfident_voter_experiment(p_eval_qkm, rbo_qm_original, correct):
    """
    Isolated toy experiment.

    We choose the weakest cyclic-mean model and sharpen only that model's
    probability distribution in a copied tensor used for entropy-weight
    computation.

    The actual aggregators still use p_eval_qkm unchanged. RBO weights are also
    fixed from the original rank-stability signal.

    Therefore, any change in entropy performance is due to entropy-weight
    sensitivity, not due to changed model votes or changed aggregation scores.
    """
    single_acc = compute_single_model_acc(p_eval_qkm, correct)
    weak_idx = int(np.argmin(single_acc))

    original_top1 = top1_signature(p_eval_qkm)
    original_ranks = rank_signature(p_eval_qkm)

    rbo_w = make_rbo_weights(rbo_qm_original, temperature=STABILITY_TEMPERATURE)

    print("\n" + "=" * 120)
    print("Isolated toy experiment: artificially overconfident weakest voter")
    print("=" * 120)
    print(f"Weakest cyclic-mean model: index={weak_idx}, model={SHORT_NAMES[weak_idx]}, acc={single_acc[weak_idx]:.4f}")
    print("Perturbation: only the entropy-weighting signal is temperature-sharpened for the weakest model.")
    print("Aggregation probabilities are unchanged. RBO weights are unchanged.")
    print("tau < 1 means stronger overconfidence. Rankings should remain unchanged.")

    diagnostic_rows = []
    rows = []

    for tau in TOY_ENTROPY_SIGNAL_TAU_GRID:
        p_entropy_signal = np.array(p_eval_qkm, copy=True)
        p_entropy_signal[:, :, weak_idx] = sharpen_probs_by_temperature(
            p_entropy_signal[:, :, weak_idx],
            tau=float(tau),
        )

        toy_top1 = top1_signature(p_entropy_signal)
        toy_ranks = rank_signature(p_entropy_signal)

        top1_changed_all_models = float(np.mean(original_top1 != toy_top1))
        full_ranking_changed_all_models = float(np.mean(np.any(original_ranks != toy_ranks, axis=2)))

        top1_changed_weak = float(np.mean(original_top1[:, weak_idx] != toy_top1[:, weak_idx]))
        full_ranking_changed_weak = float(np.mean(
            np.any(original_ranks[:, weak_idx, :] != toy_ranks[:, weak_idx, :], axis=1)
        ))

        ent_w = make_entropy_weights(p_entropy_signal, temperature=ENTROPY_TEMPERATURE)

        weak_entropy = entropy(p_entropy_signal[:, :, weak_idx], axis=1)

        diagnostic_rows.append({
            "tau": float(tau),
            "weak_model_mean_entropy": float(np.mean(weak_entropy)),
            "weak_model_entropy_weight": float(np.mean(ent_w[:, weak_idx])),
            "weak_model_rbo_weight": float(np.mean(rbo_w[:, weak_idx])),
            "top1_changed_weak_model": top1_changed_weak,
            "full_ranking_changed_weak_model": full_ranking_changed_weak,
            "top1_changed_all_models": top1_changed_all_models,
            "full_ranking_changed_all_models": full_ranking_changed_all_models,
        })

        for method in AGG_METHODS:
            # Critical: both methods aggregate unchanged p_eval_qkm.
            ent_preds, _ = aggregate_preds(p_eval_qkm, method, weights_qm=ent_w)
            rbo_preds, _ = aggregate_preds(p_eval_qkm, method, weights_qm=rbo_w)

            ent_correct = ent_preds == correct
            rbo_correct = rbo_preds == correct
            pair = compare_two_prediction_sets(ent_preds, rbo_preds, correct)

            rows.append({
                "tau": float(tau),
                "method": method,
                "weak_model_mean_entropy": float(np.mean(weak_entropy)),
                "weak_model_entropy_weight": float(np.mean(ent_w[:, weak_idx])),
                "weak_model_rbo_weight": float(np.mean(rbo_w[:, weak_idx])),
                "entropy_acc": float(np.mean(ent_correct)),
                "rbo_acc": float(np.mean(rbo_correct)),
                "entropy_minus_rbo": float(np.mean(ent_correct) - np.mean(rbo_correct)),
                "entropy_wins": pair["wins_a"],
                "rbo_wins": pair["wins_b"],
                "mcnemar_p_entropy_vs_rbo": pair["mcnemar_p"],
            })

    print_table(
        diagnostic_rows,
        "Toy diagnostics: sanity checks for isolated perturbation",
        headers=[
            "tau",
            "weak_model_mean_entropy",
            "weak_model_entropy_weight",
            "weak_model_rbo_weight",
            "top1_changed_weak_model",
            "full_ranking_changed_weak_model",
            "top1_changed_all_models",
            "full_ranking_changed_all_models",
        ],
        nd=6,
    )

    print_table(
        rows,
        "Toy overconfidence sweep: entropy vs RBO",
        headers=[
            "tau",
            "method",
            "weak_model_mean_entropy",
            "weak_model_entropy_weight",
            "weak_model_rbo_weight",
            "entropy_acc",
            "rbo_acc",
            "entropy_minus_rbo",
            "entropy_wins",
            "rbo_wins",
            "mcnemar_p_entropy_vs_rbo",
        ],
        nd=6,
    )

    compact = []
    for tau in TOY_ENTROPY_SIGNAL_TAU_GRID:
        block = [r for r in rows if abs(r["tau"] - float(tau)) < 1e-12]
        ordinal = [r for r in block if r["method"] in ORDINAL_AGG_METHODS]
        compact.append({
            "tau": float(tau),
            "weak_entropy_weight": float(np.mean([r["weak_model_entropy_weight"] for r in block])),
            "weak_rbo_weight": float(np.mean([r["weak_model_rbo_weight"] for r in block])),
            "entropy_acc_ordinal_mean": float(np.mean([r["entropy_acc"] for r in ordinal])),
            "rbo_acc_ordinal_mean": float(np.mean([r["rbo_acc"] for r in ordinal])),
            "entropy_minus_rbo_ordinal_mean": float(np.mean([r["entropy_minus_rbo"] for r in ordinal])),
        })

    print_table(
        compact,
        "Toy sweep compact summary across ordinal aggregators",
        headers=[
            "tau",
            "weak_entropy_weight",
            "weak_rbo_weight",
            "entropy_acc_ordinal_mean",
            "rbo_acc_ordinal_mean",
            "entropy_minus_rbo_ordinal_mean",
        ],
        nd=6,
    )

    return rows, compact, diagnostic_rows


# =============================================================================
# Main
# =============================================================================

def main():
    data = load_dataset_pack(N_SAMPLES, BENCHMARK, DATASET_SEED)
    correct = data.correct

    K = len(data.option_labels)
    shifts = choose_cyclic_shifts(K=K, max_shifts=MAX_CYCLIC_SHIFTS, seed=CYCLIC_SHIFT_SEED)
    if 0 not in shifts:
        raise RuntimeError("cyclic shifts must include 0")

    raw_cyclic = load_cyclic_cache(CYCLIC_NPZ_FILE, data, shifts)

    print("\n" + "=" * 120)
    print("Entropy vs RBO-rank comparison")
    print("=" * 120)
    print(f"N={len(correct)} | K={K} | M={len(MODEL_NAMES)} | benchmark={data.benchmark_resolved}")
    print(f"Cache: {CYCLIC_NPZ_FILE}")
    print(f"RBO p={RBO_P} | stability softmax T={STABILITY_TEMPERATURE} | entropy softmax T={ENTROPY_TEMPERATURE}")

    p_cyc_qkms = get_cyclic_dists(raw_cyclic)
    p_cycmean_qkm = pool_cyclic_per_model(p_cyc_qkms)

    print("\nComputing RBO stability features ...")
    rbo_qm = compute_rbo_features(p_cyc_qkms, p=RBO_P)

    paired_entropy_vs_rbo_table(
        p_qkm=p_cycmean_qkm,
        rbo_qm=rbo_qm,
        correct=correct,
        title="Direct paired comparison: entropy weighting vs RBO-rank weighting",
    )

    toy_overconfident_voter_experiment(
        p_eval_qkm=p_cycmean_qkm,
        rbo_qm_original=rbo_qm,
        correct=correct,
    )


if __name__ == "__main__":
    main()