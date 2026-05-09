import gc
import json
import logging
import math
import os
import re
import hashlib
import csv
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
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import transformers
import datasets as hf_datasets

try:
    from transformers import Gemma3ForConditionalGeneration
except Exception:
    Gemma3ForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(level=logging.ERROR)
for _l in [
    "transformers", "datasets", "huggingface_hub", "huggingface_hub.utils",
    "httpx", "httpcore", "urllib3", "filelock", "fsspec", "PIL"
]:
    logging.getLogger(_l).setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
hf_datasets.logging.set_verbosity_error()


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAMES = [
    #"Qwen/Qwen2.5-32B-Instruct",
    #"Qwen/Qwen3-32B",
    #"mistralai/Mistral-Small-24B-Instruct-2501",
    #"meta-llama/Llama-3.1-70B-Instruct",
    #"google/gemma-2-27b-it",
    # "microsoft/Phi-4-mini-instruct",
    # "ibm-granite/granite-3.3-2b-instruct",
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

BENCHMARK = "mmlu"
N_SAMPLES = 14042
DATASET_SEED = 42
USE_CACHE = True
SCORE_BATCH = 64
MAX_LENGTH = 2048
EPS = 1e-6

DATASET_K_FILTER: Union[str, int] = "modal"
INCLUDE_PASSAGE_IN_QUESTION = True
RACE_CONFIG = "all"
GPQA_SHUFFLE_OPTIONS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

OUT_DIR = "./majority_regime_analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_CACHE_DIR = os.path.join(OUT_DIR, "per_model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Keep these names unchanged so old caches are reused.
CACHE_TAG = f"{BENCHMARK}_K{DATASET_K_FILTER}_N{N_SAMPLES}_seed{DATASET_SEED}_M{len(MODEL_NAMES)}"
GEN_NPZ_FILE = os.path.join(OUT_DIR, f"gen_scores_{CACHE_TAG}.npz")

RUN_CYCLIC_SHIFT = True
MAX_CYCLIC_SHIFTS = None
CYCLIC_SHIFT_SEED = 123
CYCLIC_TAG = (
    f"{CACHE_TAG}_cyclic"
    f"_S{'all' if MAX_CYCLIC_SHIFTS is None else MAX_CYCLIC_SHIFTS}"
    f"_shiftseed{CYCLIC_SHIFT_SEED}"
)
CYCLIC_NPZ_FILE = os.path.join(OUT_DIR, f"cyclic_gen_scores_{CYCLIC_TAG}.npz")
JSON_FILE = os.path.join(OUT_DIR, f"paper_facing_stability_eval_{CACHE_TAG}.json")

V_CORRECT = 0

ORDINAL_AGG_METHODS = ["hard_majority", "borda", "mrr", "irv"]
PROB_AGG_METHODS = ["arithmetic_mean", "geometric_mean"]
NAIVE_AGG_METHODS = ORDINAL_AGG_METHODS + PROB_AGG_METHODS
PRIMARY_NAIVE_BASELINE = "arithmetic_mean"

REPORT_ENTROPY_WEIGHTING = True
PAPER_STABILITY_FEATURES = ["rbo_p85", "rank_rr_overlap_alpha2"]
PAPER_STABILITY_TEMPERATURE = 1.0
ENTROPY_WEIGHT_TEMPERATURE = 1.0

# Main-paper default is still rbo_p85 at T=1.0 via PAPER_STABILITY_FEATURES and
# PAPER_STABILITY_TEMPERATURE. The expanded grid below is only for post-hoc
# sensitivity analysis and does not affect cache paths or model scoring.
RBO_PS = [0.85]
SENSITIVITY_TEMPERATURES = [1.0]
SENSITIVITY_METHODS = ORDINAL_AGG_METHODS
OVERLAP_AT_K = [1, 2, 3, 5]
BOOTSTRAP_SEED = 12345
BOOTSTRAP_N = 5000
STDOUT_TOP_ROWS = 200

# If True, the compact printed table always selects rbo_p85 variants instead of
# taking the best among all stability signals. This matches the current paper story.
COMPACT_TABLE_FORCE_SIGNAL = "rbo_p85"


# =============================================================================
# Prompt
# =============================================================================

C_GEN_TMPL = (
    "Answer the following single-choice question.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Select the correct option. Answer with only the option letter.\n"
    "Answer:"
)


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


def canonical_benchmark_name(benchmark):
    b = benchmark.strip().lower().replace("-", "_")
    aliases = {
        "mmlupro": "mmlu_pro", "mmlu_pro": "mmlu_pro", "mmlu": "mmlu",
        "arc_easy": "arc_easy", "arceasy": "arc_easy", "arc_e": "arc_easy",
        "arc_challenge": "arc_challenge", "arcchallenge": "arc_challenge", "arc_c": "arc_challenge",
        "race": "race", "race_all": "race", "race_high": "race_high", "race_middle": "race_middle",
        "gpqa": "gpqa_diamond", "gpqa_diamond": "gpqa_diamond", "gpqa_main": "gpqa_main",
        "gpqa_extended": "gpqa_extended", "gpqa_experts": "gpqa_experts",
        "medqa": "medqa", "openlifescienceai_medqa": "medqa",
    }
    if b not in aliases:
        raise ValueError(f"Unknown BENCHMARK={benchmark!r}")
    return aliases[b]


def stable_json_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", s)


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
    if benchmark in ["mmlu", "race", "race_high", "race_middle", "gpqa_diamond", "gpqa_main", "gpqa_extended", "gpqa_experts", "medqa"]:
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

    option_lengths = [len(o) for _, _, o, _ in examples]
    target_k, k_filter_name = choose_target_k(option_lengths, benchmark)
    filtered = [(i, q, o, a) for (i, q, o, a) in examples if len(o) == target_k and 0 <= int(a) < target_k]
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


def fmt_opts(labels, opts):
    return "".join(f"{l}) {o}\n" for l, o in zip(labels, opts))


# =============================================================================
# Model scoring
# =============================================================================

def unique_model_first_indices(model_names):
    first = {}
    ordered_first_indices = []
    for i, name in enumerate(model_names):
        if name not in first:
            first[name] = i
            ordered_first_indices.append(i)
    return first, ordered_first_indices


def is_gemma3_model(name: str) -> bool:
    return "gemma-3" in name.lower() or "gemma3" in name.lower()


def is_gemma3_text_only_model(name: str) -> bool:
    n = name.lower()
    return is_gemma3_model(name) and ("1b" in n or "270m" in n)


def get_model_input_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def load_model(name):
    print(f"Loading {name} ...")
    processor = None

    if is_gemma3_model(name) and not is_gemma3_text_only_model(name):
        if Gemma3ForConditionalGeneration is None and AutoModelForImageTextToText is None:
            raise RuntimeError(
                f"{name} looks like a Gemma 3 VLM checkpoint, but this transformers "
                "installation has neither Gemma3ForConditionalGeneration nor "
                "AutoModelForImageTextToText. Upgrade transformers or use google/gemma-3-1b-it."
            )
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        tok = getattr(processor, "tokenizer", None)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model_cls = Gemma3ForConditionalGeneration if Gemma3ForConditionalGeneration is not None else AutoModelForImageTextToText
        model = model_cls.from_pretrained(
            name,
            torch_dtype=DTYPE,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    else:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=DTYPE,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    model.eval()
    model._hf_model_name_for_scoring = name
    model._hf_processor_for_scoring = processor
    return model, tok


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_chat(tok, raw, model_name: str = ""):
    if not getattr(tok, "chat_template", None):
        return raw

    messages = [{"role": "user", "content": raw}]
    lower_name = model_name.lower()

    if "qwen3" in lower_name:
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass

    if "granite-3.3" in lower_name:
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking=False,
            )
        except TypeError:
            pass

    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def single_token_ids(tok, forms):
    ids = set()
    for f in forms:
        t = tok.encode(f, add_special_tokens=False)
        if len(t) == 1:
            ids.add(int(t[0]))
    return torch.tensor(sorted(ids), dtype=torch.long)


def label_token_groups(tok, labels):
    groups = []
    for lab in labels:
        g = single_token_ids(tok, [lab, f" {lab}"])
        assert len(g) > 0, f"No single-token form for label {lab!r}"
        groups.append(g)
    return groups


@torch.inference_mode()
def next_token_logprobs(model, tok, prompts, groups, batch_size=SCORE_BATCH):
    out = np.full((len(prompts), len(groups)), -np.inf, dtype=np.float64)
    input_device = get_model_input_device(model)
    groups = [g.to(input_device) for g in groups]
    processor = getattr(model, "_hf_processor_for_scoring", None)
    n_trunc = 0

    for s in range(0, len(prompts), batch_size):
        bp = list(prompts[s:s + batch_size])
        lengths = [len(x) for x in tok(bp, return_tensors=None, add_special_tokens=False)["input_ids"]]
        if max(lengths) > MAX_LENGTH:
            n_trunc += 1
            if n_trunc <= 3:
                print(f"WARNING: truncating batch {s // batch_size}; max={max(lengths)}; MAX_LENGTH={MAX_LENGTH}")

        if processor is not None:
            try:
                enc = processor(text=bp, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            except TypeError:
                enc = tok(bp, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
        else:
            enc = tok(bp, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)

        enc = move_batch_to_device(enc, input_device)
        outputs = model(**enc)
        logits = outputs.logits

        if tok.padding_side == "left":
            last = logits.shape[1] - 1
        else:
            last = enc["attention_mask"].sum(dim=1) - 1

        bidx = torch.arange(logits.shape[0], device=logits.device)
        lp = torch.log_softmax(logits[bidx, last, :].float(), dim=-1)
        arr = torch.stack([torch.logsumexp(lp[:, g.to(logits.device)], dim=-1) for g in groups], dim=1)
        out[s:s + len(bp)] = arr.detach().cpu().numpy()

        del enc, outputs, logits, lp, arr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if n_trunc > 3:
        print(f"WARNING: {n_trunc} truncated batches total")
    return out


# =============================================================================
# Numerical helpers
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


def rank_positions(scores, axis=-1):
    """
    Vectorized 1-based ranks along `axis`.
    Highest score gets rank 1.
    """
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


def rank_of_index(scores, idx):
    return int(rank_positions(scores)[int(idx)])


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


def auroc_score(y_true, score):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0 or len(y) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(y), dtype=np.float64)
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def option_level_auroc(scores_qk, correct):
    Q, K = scores_qk.shape
    y = np.zeros(Q * K, dtype=np.int64)
    s = np.asarray(scores_qk, dtype=np.float64).reshape(-1)
    y[np.arange(Q) * K + correct.astype(np.int64)] = 1
    return auroc_score(y, s)


def fmt_float(x, nd=4):
    try:
        if np.isnan(x):
            return "nan"
    except Exception:
        pass
    return f"{float(x):.{nd}f}"


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
    return obj


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


def bootstrap_delta_ci_p(preds_a, preds_b, correct, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    a = (np.asarray(preds_a) == correct).astype(np.float64)
    b = (np.asarray(preds_b) == correct).astype(np.float64)
    d = a - b
    n = len(d)
    obs = float(np.mean(d))
    if n <= 1:
        return {"bootstrap_delta": obs, "bootstrap_ci95_low": float("nan"), "bootstrap_ci95_high": float("nan"), "bootstrap_p_two_sided": float("nan")}
    vals = np.empty(n_boot, dtype=np.float64)
    for t in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[t] = np.mean(d[idx])
    p_left = np.mean(vals <= 0.0)
    p_right = np.mean(vals >= 0.0)
    p_two = float(min(1.0, 2.0 * min(p_left, p_right)))
    return {
        "bootstrap_delta": obs,
        "bootstrap_ci95_low": float(np.quantile(vals, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(vals, 0.975)),
        "bootstrap_p_two_sided": p_two,
    }


# =============================================================================
# Cache handling and scoring
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


def cyclic_permutation_indices(K, shift):
    return [int((j + shift) % K) for j in range(K)]


def model_cache_path(model_name: str, data: DatasetPack, shifts: List[int]) -> str:
    tag_obj = {
        "model": model_name,
        "benchmark": data.benchmark_resolved,
        "dataset_seed": DATASET_SEED,
        "k_filter": data.k_filter,
        "K": len(data.option_labels),
        "include_passage": INCLUDE_PASSAGE_IN_QUESTION,
        "race_config": RACE_CONFIG,
        "gpqa_shuffle": GPQA_SHUFFLE_OPTIONS,
        "cyclic_shifts": list(map(int, shifts)),
        "prompt_hash": stable_json_hash(C_GEN_TMPL),
        "max_length": MAX_LENGTH,
        "score_type": "next_token_label_logprobs_cyclic",
        "cache_version": 2,
    }
    h = stable_json_hash(tag_obj)[:12]
    return os.path.join(
        MODEL_CACHE_DIR,
        f"{safe_filename(data.benchmark_resolved)}__seed{DATASET_SEED}__K{len(data.option_labels)}__{safe_filename(model_name)}__{h}.npz",
    )


def load_per_model_cache(path: str, data: DatasetPack, shifts: List[int]) -> Dict[str, Any]:
    empty = {"question_ids": [], "source_indices": np.asarray([], dtype=np.int64), "raw_cyclic": None, "metadata": {}}
    if not (USE_CACHE and os.path.exists(path)):
        return empty
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"Per-model cache load error for {path}: {e}")
        return empty
    required = ["question_ids", "raw_cyclic", "source_indices", "shifts", "option_labels"]
    if any(k not in npz.files for k in required):
        print(f"Per-model cache missing fields, ignoring: {path}")
        return empty
    if list(npz["shifts"].astype(int).tolist()) != list(map(int, shifts)):
        print(f"Per-model cache shifts differ, ignoring: {path}")
        return empty
    if list(npz["option_labels"].tolist()) != data.option_labels:
        print(f"Per-model cache option labels differ, ignoring: {path}")
        return empty
    raw = npz["raw_cyclic"]
    if raw.ndim != 3 or raw.shape[1] != len(data.option_labels) or raw.shape[2] != len(shifts):
        print(f"Per-model cache shape mismatch, ignoring: {path}: {raw.shape}")
        return empty
    metadata = {}
    if "metadata_json" in npz.files:
        try:
            metadata = json.loads(str(npz["metadata_json"].tolist()))
        except Exception:
            metadata = {}
    return {
        "question_ids": [str(x) for x in npz["question_ids"].tolist()],
        "source_indices": npz["source_indices"].astype(np.int64),
        "raw_cyclic": raw.astype(np.float64),
        "metadata": metadata,
    }


def merge_cache_records(old: Dict[str, Any], new_qids: List[str], new_source_indices: List[int], new_raw: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
    records: Dict[str, Tuple[int, np.ndarray]] = {}
    if old.get("raw_cyclic") is not None:
        for qid, si, row in zip(old["question_ids"], old["source_indices"].tolist(), old["raw_cyclic"]):
            records[str(qid)] = (int(si), np.asarray(row, dtype=np.float64))
    for qid, si, row in zip(new_qids, new_source_indices, new_raw):
        records[str(qid)] = (int(si), np.asarray(row, dtype=np.float64))
    qids = sorted(records.keys())
    source_indices = np.asarray([records[qid][0] for qid in qids], dtype=np.int64)
    raw = np.stack([records[qid][1] for qid in qids], axis=0).astype(np.float64)
    return qids, source_indices, raw


def save_per_model_cache(path: str, model_name: str, data: DatasetPack, shifts: List[int], qids: List[str], source_indices: np.ndarray, raw: np.ndarray):
    metadata = {
        "model_name": model_name,
        "benchmark_resolved": data.benchmark_resolved,
        "dataset_seed": DATASET_SEED,
        "k_filter": data.k_filter,
        "K": len(data.option_labels),
        "n_cached_questions": int(len(qids)),
        "cyclic_shifts": list(map(int, shifts)),
        "prompt_template_hash": stable_json_hash(C_GEN_TMPL),
        "cache_version": 2,
    }
    tmp = path + ".tmp"
    np.savez_compressed(
        tmp,
        question_ids=np.asarray(qids, dtype=object),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        raw_cyclic=np.asarray(raw, dtype=np.float64),
        shifts=np.asarray(shifts, dtype=np.int64),
        option_labels=np.asarray(data.option_labels, dtype=object),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
    )
    tmp_npz = tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp
    os.replace(tmp_npz, path)
    print(f"Saved per-model cache: {path} | n={len(qids)}")


@torch.inference_mode()
def score_model_cyclic_shifts_subset(model, tok, data: DatasetPack, q_indices: List[int], shifts: List[int], model_name) -> np.ndarray:
    K, S = len(data.option_labels), len(shifts)
    lab_g = label_token_groups(tok, data.option_labels)
    raw_cyclic = np.zeros((len(q_indices), K, S), dtype=np.float64)
    prompts, meta = [], []
    for local_idx, qi in enumerate(q_indices):
        for si, shift in enumerate(shifts):
            perm = cyclic_permutation_indices(K, shift)
            raw = C_GEN_TMPL.format(
                question=data.questions[qi],
                options=fmt_opts(data.option_labels, [data.options[qi][p] for p in perm]),
            )
            prompts.append(apply_chat(tok, raw, model_name=model_name))
            meta.append((local_idx, si, perm))
    lp = next_token_logprobs(model, tok, prompts, lab_g)
    for row, (local_idx, si, perm) in zip(lp, meta):
        for displayed_j, original_k in enumerate(perm):
            raw_cyclic[local_idx, original_k, si] = row[displayed_j]
    return raw_cyclic


def score_or_load_model_cyclic(model_name: str, data: DatasetPack, shifts: List[int]) -> np.ndarray:
    Q, K, S = len(data.questions), len(data.option_labels), len(shifts)
    raw_one = np.zeros((Q, K, S), dtype=np.float64)
    path = model_cache_path(model_name, data, shifts)
    cache = load_per_model_cache(path, data, shifts)
    id_to_cached_idx = {qid: idx for idx, qid in enumerate(cache["question_ids"])}
    missing_qidx = []
    filled_from_cache = 0
    for qi, qid in enumerate(data.question_ids):
        if qid in id_to_cached_idx and cache.get("raw_cyclic") is not None:
            raw_one[qi] = cache["raw_cyclic"][id_to_cached_idx[qid]]
            filled_from_cache += 1
        else:
            missing_qidx.append(qi)
    print(f"Model cache status | {model_name.split('/')[-1]} | cached={filled_from_cache}/{Q} | missing={len(missing_qidx)} | path={path}")
    if missing_qidx:
        model, tok = load_model(model_name)
        new_raw = score_model_cyclic_shifts_subset(model, tok, data, missing_qidx, shifts, model_name)
        unload_model(model)
        for local_idx, qi in enumerate(missing_qidx):
            raw_one[qi] = new_raw[local_idx]
        merged_qids, merged_source_indices, merged_raw = merge_cache_records(
            cache,
            [data.question_ids[qi] for qi in missing_qidx],
            [data.source_indices[qi] for qi in missing_qidx],
            new_raw,
        )
        save_per_model_cache(path, model_name, data, shifts, merged_qids, merged_source_indices, merged_raw)
    else:
        print(f"No inference needed for {model_name.split('/')[-1]}.")
    return raw_one


def load_cyclic_cache(path, data, shifts):
    if not os.path.exists(path):
        return None
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"Cyclic run-cache load error: {e}")
        return None
    Q, K, M, S = len(data.questions), len(data.option_labels), len(MODEL_NAMES), len(shifts)
    if "raw_cyclic" not in npz.files:
        return None
    rc = npz["raw_cyclic"]
    if rc.shape != (Q, K, M, S):
        print(f"Cyclic run-cache shape mismatch: {rc.shape}; expected {(Q, K, M, S)}")
        return None
    if "shifts" in npz.files and list(npz["shifts"].astype(int).tolist()) != list(shifts):
        print("Cyclic run-cache shifts differ")
        return None
    if "model_names" in npz.files and list(npz["model_names"].tolist()) != MODEL_NAMES:
        print("Cyclic run-cache model list differs")
        return None
    if "correct" in npz.files and not np.array_equal(npz["correct"].astype(np.int64), data.correct):
        print("Cyclic run-cache correct labels differ")
        return None
    print(f"Loaded full cyclic run-cache: {path}")
    return rc


def save_cyclic_cache(path, data, raw_cyclic, shifts):
    np.savez_compressed(
        path,
        raw_cyclic=raw_cyclic,
        shifts=np.asarray(shifts, dtype=np.int64),
        correct=data.correct,
        source_indices=np.asarray(data.source_indices, dtype=np.int64),
        model_names=np.asarray(MODEL_NAMES, dtype=object),
        option_labels=np.asarray(data.option_labels, dtype=object),
    )
    print(f"Saved full cyclic run-cache: {path}")


def save_gen_cache(path, data, raw_gen):
    np.savez_compressed(
        path,
        raw_gen=raw_gen,
        correct=data.correct,
        source_indices=np.asarray(data.source_indices, dtype=np.int64),
        model_names=np.asarray(MODEL_NAMES, dtype=object),
        option_labels=np.asarray(data.option_labels, dtype=object),
        benchmark_resolved=np.asarray([data.benchmark_resolved], dtype=object),
    )
    print(f"Saved generator cache: {path}")


def derive_raw_gen_from_cyclic(raw_cyclic, shifts):
    if 0 not in list(shifts):
        raise ValueError("Cannot derive raw_gen from cyclic cache because shift 0 is absent.")
    si0 = list(shifts).index(0)
    return raw_cyclic[:, :, :, si0][:, :, :, None, None].copy()


def run_or_load_all_scores(data, shifts):
    Q, K, M, S = len(data.questions), len(data.option_labels), len(MODEL_NAMES), len(shifts)
    full_cache = load_cyclic_cache(CYCLIC_NPZ_FILE, data, shifts) if USE_CACHE else None
    if full_cache is not None:
        raw_cyclic = full_cache
        raw_gen = derive_raw_gen_from_cyclic(raw_cyclic, shifts)
        return raw_gen, raw_cyclic

    raw_cyclic = np.zeros((Q, K, M, S), dtype=np.float64)
    _, ordered_first_indices = unique_model_first_indices(MODEL_NAMES)
    if len(ordered_first_indices) < M:
        n_dupes = M - len(ordered_first_indices)
        print(f"Detected {n_dupes} duplicate MODEL_NAMES entries; duplicate scores copied without reloading.")
    model_name_to_scores = {}
    for first_mi in ordered_first_indices:
        mn = MODEL_NAMES[first_mi]
        model_name_to_scores[mn] = score_or_load_model_cyclic(mn, data, shifts)
    for mi, mn in enumerate(MODEL_NAMES):
        raw_cyclic[:, :, mi, :] = model_name_to_scores[mn]
    if USE_CACHE:
        save_cyclic_cache(CYCLIC_NPZ_FILE, data, raw_cyclic, shifts)
    raw_gen = derive_raw_gen_from_cyclic(raw_cyclic, shifts)
    if USE_CACHE:
        save_gen_cache(GEN_NPZ_FILE, data, raw_gen)
    return raw_gen, raw_cyclic


def get_mc_logits(raw_gen):
    return raw_gen[:, :, :, 0, V_CORRECT]


def get_mc_dists(raw_gen):
    return np.exp(logsoftmax(get_mc_logits(raw_gen), axis=1))


def get_cyclic_dists(raw_cyclic):
    return np.exp(logsoftmax(raw_cyclic, axis=1))


def pool_cyclic_per_model(p_cyc_qkms, method="mean"):
    if method == "mean":
        return normalize(np.mean(p_cyc_qkms, axis=3), axis=1)
    if method == "geom":
        return np.exp(logsoftmax(np.mean(np.log(np.maximum(p_cyc_qkms, EPS)), axis=3), axis=1))
    raise ValueError(method)


# =============================================================================
# Vectorized cyclic stability features
# =============================================================================

def rbo_signal_name(p: float) -> str:
    return f"rbo_p{int(round(100 * float(p))):02d}"


def cyclic_stability_features(p_cyc_qkms, p_cycmean_qkm, shifts):
    """
    Vectorized cyclic stability feature computation.

    p_cyc_qkms: shape (Q, K, M, S)
    p_cycmean_qkm: shape (Q, K, M)
    """
    p_cyc_qkms = np.asarray(p_cyc_qkms, dtype=np.float64)
    p_cycmean_qkm = np.asarray(p_cycmean_qkm, dtype=np.float64)
    Q, K, M, S = p_cyc_qkms.shape

    rbo_feature_names = [rbo_signal_name(float(p)) for p in RBO_PS]
    feature_names = [
        "top1_agreement",
        "top1_entropy_stability",
        "winner_mrr_stability",
        "winner_rr2_stability",
        "rank_rr_overlap_alpha2",
        "cyclic_mean_top_prob",
        "cyclic_mean_margin",
        "cyclic_mean_neg_entropy",
    ] + rbo_feature_names + [f"overlap_at_{k}" for k in OVERLAP_AT_K]
    feats = {name: np.zeros((Q, M), dtype=np.float64) for name in feature_names}

    p_bar_qkm = normalize(p_cycmean_qkm, axis=1)
    sorted_bar_qkm = np.sort(p_bar_qkm, axis=1)[:, ::-1, :]
    feats["cyclic_mean_top_prob"] = sorted_bar_qkm[:, 0, :]
    feats["cyclic_mean_margin"] = sorted_bar_qkm[:, 0, :] - sorted_bar_qkm[:, 1, :] if K > 1 else sorted_bar_qkm[:, 0, :]
    feats["cyclic_mean_neg_entropy"] = -entropy(np.transpose(p_bar_qkm, (0, 2, 1)), axis=2)

    tops_qms = np.argmax(p_cyc_qkms, axis=1)
    top_counts_qmk = np.zeros((Q, M, K), dtype=np.float64)
    q_idx = np.repeat(np.arange(Q), M * S)
    m_idx = np.tile(np.repeat(np.arange(M), S), Q)
    k_idx = tops_qms.reshape(-1)
    np.add.at(top_counts_qmk, (q_idx, m_idx, k_idx), 1.0)
    top_dist_qmk = normalize(top_counts_qmk, axis=2)
    feats["top1_agreement"] = np.max(top_counts_qmk, axis=2) / max(S, 1)
    feats["top1_entropy_stability"] = 1.0 - entropy(top_dist_qmk, axis=2) / max(np.log(K), EPS)

    ranks_qkms = rank_positions(p_cyc_qkms, axis=1).astype(np.float64)
    winner_qm = np.argmax(p_bar_qkm, axis=1)
    winner_ranks_qms = np.take_along_axis(
        np.transpose(ranks_qkms, (0, 2, 3, 1)),
        winner_qm[:, :, None, None],
        axis=3,
    )[:, :, :, 0]
    feats["winner_mrr_stability"] = np.mean(1.0 / winner_ranks_qms, axis=2)
    feats["winner_rr2_stability"] = np.mean(1.0 / np.square(winner_ranks_qms), axis=2)

    alpha = 2.0
    v_qkms = 1.0 / np.power(ranks_qkms, alpha)
    norm_qms = np.linalg.norm(v_qkms, axis=1, keepdims=True)
    v_qkms = v_qkms / np.maximum(norm_qms, EPS)
    pair_vals = []
    for s1 in range(S):
        for s2 in range(s1 + 1, S):
            pair_vals.append(np.sum(v_qkms[:, :, :, s1] * v_qkms[:, :, :, s2], axis=1))
    feats["rank_rr_overlap_alpha2"] = np.mean(np.stack(pair_vals, axis=2), axis=2) if pair_vals else np.ones((Q, M), dtype=np.float64)

    orders_qkms = np.argsort(-p_cyc_qkms, axis=1, kind="stable")

    prefix_masks = np.zeros((K, Q, M, S, K), dtype=bool)
    qq = np.repeat(np.arange(Q), M * S)
    mm = np.tile(np.repeat(np.arange(M), S), Q)
    ss = np.tile(np.arange(S), Q * M)

    for d in range(K):
        mask = np.zeros((Q, M, S, K), dtype=bool)
        for jj in range(d + 1):
            chosen = orders_qkms[:, jj, :, :]
            mask[qq, mm, ss, chosen.reshape(-1)] = True
        prefix_masks[d] = mask

    rbo_pair_vals_by_p = {float(p): [] for p in RBO_PS}
    overlap_pair_vals_by_k = {topk: [] for topk in OVERLAP_AT_K}
    rbo_weights_by_p = {
        float(p): np.asarray([(1.0 - float(p)) * (float(p) ** d) for d in range(K)], dtype=np.float64)
        for p in RBO_PS
    }

    for s1 in range(S):
        for s2 in range(s1 + 1, S):
            overlap_by_depth = []
            for d in range(K):
                inter = np.sum(prefix_masks[d, :, :, s1, :] & prefix_masks[d, :, :, s2, :], axis=2)
                overlap_by_depth.append(inter / float(d + 1))

            final_inter = np.sum(prefix_masks[K - 1, :, :, s1, :] & prefix_masks[K - 1, :, :, s2, :], axis=2)
            final_overlap = final_inter / float(K)

            for p_rbo, rbo_weights in rbo_weights_by_p.items():
                rbo_val = np.zeros((Q, M), dtype=np.float64)
                for d in range(K):
                    rbo_val += rbo_weights[d] * overlap_by_depth[d]
                rbo_val += (p_rbo ** K) * final_overlap
                rbo_pair_vals_by_p[p_rbo].append(rbo_val)

            for topk in OVERLAP_AT_K:
                kk = min(int(topk), K)
                inter = np.sum(prefix_masks[kk - 1, :, :, s1, :] & prefix_masks[kk - 1, :, :, s2, :], axis=2)
                overlap_pair_vals_by_k[topk].append(inter / float(max(kk, 1)))

    for p_rbo, vals in rbo_pair_vals_by_p.items():
        name = rbo_signal_name(float(p_rbo))
        feats[name] = np.mean(np.stack(vals, axis=2), axis=2) if vals else np.ones((Q, M), dtype=np.float64)

    for topk in OVERLAP_AT_K:
        vals = overlap_pair_vals_by_k[topk]
        feats[f"overlap_at_{topk}"] = np.mean(np.stack(vals, axis=2), axis=2) if vals else np.ones((Q, M), dtype=np.float64)

    return feats


# =============================================================================
# Vectorized aggregation
# =============================================================================

def weighted_vote_counts(pred_qm, weights_qm, K):
    pred_qm = np.asarray(pred_qm, dtype=np.int64)
    weights_qm = np.asarray(weights_qm, dtype=np.float64)
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
    order_qmk = np.asarray(order_qmk, dtype=np.int64)
    weights_qm = np.asarray(weights_qm, dtype=np.float64)
    Q, M, _ = order_qmk.shape
    scores = np.zeros((Q, K), dtype=np.float64)
    pos_scores = (K - 1 - np.arange(K, dtype=np.float64))
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

    # Kept loop-based because IRV has sequential elimination. K is usually 4,
    # so this is not the dominant cost after vectorizing stability/Borda/MRR.
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
    p_qkm = np.asarray(p_qkm, dtype=np.float64)
    Q, K, M = p_qkm.shape
    if weights_qm is None:
        weights_qm = np.ones((Q, M), dtype=np.float64)
    else:
        weights_qm = np.asarray(weights_qm, dtype=np.float64)

    if method == "hard_majority":
        pred_qm = np.argmax(p_qkm, axis=1)
        return weighted_vote_counts(pred_qm, weights_qm, K)

    if method == "irv":
        return instant_runoff_scores(p_qkm, weights_qm=weights_qm)

    if method == "arithmetic_mean":
        denom = np.maximum(np.sum(weights_qm, axis=1, keepdims=True), EPS)
        return np.sum(p_qkm * weights_qm[:, None, :], axis=2) / denom

    if method == "geometric_mean":
        denom = np.maximum(np.sum(weights_qm, axis=1, keepdims=True), EPS)
        logp = np.log(np.maximum(p_qkm, EPS))
        return np.sum(logp * weights_qm[:, None, :], axis=2) / denom

    if method in ["borda", "mrr"]:
        ranks_qkm = rank_positions(p_qkm, axis=1).astype(np.float64)
        if method == "borda":
            contrib_qkm = K - ranks_qkm
        else:
            contrib_qkm = 1.0 / ranks_qkm
        return np.sum(contrib_qkm * weights_qm[:, None, :], axis=2)

    raise ValueError(method)


def aggregate_preds(p_qkm, method, weights_qm=None):
    scores = aggregate_scores(p_qkm, method, weights_qm=weights_qm)
    return np.argmax(scores, axis=1).astype(np.int64), scores


def top_margin_from_scores(scores_qk):
    sorted_scores = np.sort(np.asarray(scores_qk, dtype=np.float64), axis=1)[:, ::-1]
    top = sorted_scores[:, 0]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1] if scores_qk.shape[1] > 1 else sorted_scores[:, 0]
    return top, margin


def make_entropy_weighted_model_weights(p_qkm, temperature=1.0):
    H_qm = entropy(np.transpose(p_qkm, (0, 2, 1)), axis=2)
    conf_qm = -H_qm
    return softmax_weights(zscore_per_question(conf_qm), temperature=temperature)


# =============================================================================
# Sensitivity analysis
# =============================================================================

def sensitivity_rows_rbo_temperature(features_qm, p_qkm, correct):
    """
    Post-hoc sweep over RBO persistence p and reliability softmax temperature T.

    This does not rescore models and does not change cache behavior. It uses
    already-computed cyclic rankings and cyclic-mean probabilities.
    """
    rows = []

    base_preds = {}
    base_acc = {}
    for method in SENSITIVITY_METHODS:
        preds, _ = aggregate_preds(p_qkm, method, weights_qm=None)
        base_preds[method] = preds
        base_acc[method] = float(np.mean(preds == correct))

    for p_rbo in RBO_PS:
        signal = rbo_signal_name(float(p_rbo))
        if signal not in features_qm:
            raise KeyError(f"Missing sensitivity signal {signal!r}; available={sorted(features_qm.keys())}")

        rel = zscore_per_question(np.asarray(features_qm[signal], dtype=np.float64))

        for T in SENSITIVITY_TEMPERATURES:
            w = softmax_weights(rel, temperature=float(T))
            for method in SENSITIVITY_METHODS:
                preds, scores = aggregate_preds(p_qkm, method, weights_qm=w)
                acc = float(np.mean(preds == correct))
                delta = acc - base_acc[method]
                base = base_preds[method]
                base_correct = base == correct
                method_correct = preds == correct
                wins = int(np.sum(method_correct & (~base_correct)))
                losses = int(np.sum((~method_correct) & base_correct))
                rows.append({
                    "method": method,
                    "p": float(p_rbo),
                    "temperature": float(T),
                    "signal": signal,
                    "base_acc": base_acc[method],
                    "acc": acc,
                    "delta": delta,
                    "wins": wins,
                    "losses": losses,
                    "mcnemar_p": mcnemar_exact_p(wins, losses),
                    "option_auroc": option_level_auroc(scores, correct),
                })

    return rows


def save_sensitivity_csvs(rows, out_dir, cache_tag):
    all_path = os.path.join(out_dir, f"rbo_sensitivity_all_{cache_tag}.csv")
    fieldnames = [
        "method", "p", "temperature", "signal", "base_acc", "acc", "delta",
        "wins", "losses", "mcnemar_p", "option_auroc",
    ]
    with open(all_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (x["method"], x["temperature"], x["p"])):
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved sensitivity CSV: {all_path}")

    method_paths = {"all": all_path}
    for method in SENSITIVITY_METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        method_path = os.path.join(out_dir, f"rbo_sensitivity_{method}_{cache_tag}.csv")
        with open(method_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["p", "temperature", "acc", "delta"])
            writer.writeheader()
            for r in sorted(method_rows, key=lambda x: (x["temperature"], x["p"])):
                writer.writerow({
                    "p": r["p"],
                    "temperature": r["temperature"],
                    "acc": r["acc"],
                    "delta": r["delta"],
                })
        method_paths[method] = method_path
        print(f"Saved sensitivity CSV for {method}: {method_path}")

    return method_paths


def print_sensitivity_table(rows):
    compact = []
    for method in SENSITIVITY_METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        best = max(method_rows, key=lambda r: r["acc"])
        default_rows = [
            r for r in method_rows
            if abs(r["p"] - 0.85) < 1e-12 and abs(r["temperature"] - 1.0) < 1e-12
        ]
        default = default_rows[0] if default_rows else None
        compact.append({
            "method": method,
            "base_acc": best["base_acc"],
            "default_acc": default["acc"] if default else float("nan"),
            "default_delta": default["delta"] if default else float("nan"),
            "best_acc": best["acc"],
            "best_delta": best["delta"],
            "best_p": best["p"],
            "best_T": best["temperature"],
        })

    print_table(
        compact,
        "RBO-p / temperature sensitivity summary",
        headers=["method", "base_acc", "default_acc", "default_delta", "best_acc", "best_delta", "best_p", "best_T"],
    )


# =============================================================================
# Paper-facing analyses
# =============================================================================

def single_model_rows(p_raw_qkm, p_cycmean_qkm, correct):
    rows = []
    for setting, p in [("raw", p_raw_qkm), ("cyclic_mean", p_cycmean_qkm)]:
        pred_qm = np.argmax(p, axis=1)
        for mi, model in enumerate(SHORT_NAMES):
            cm = pred_qm[:, mi] == correct
            rows.append({
                "setting": setting,
                "model": model,
                "n": int(len(correct)),
                "acc": float(np.mean(cm)),
                "correct": int(np.sum(cm)),
                "incorrect": int(np.sum(~cm)),
            })
    return rows


def oracle_reference_rows(p_qkm, correct):
    pred_qm = np.argmax(p_qkm, axis=1)
    model_acc = np.mean(pred_qm == correct[:, None], axis=0)
    best_m = int(np.argmax(model_acc))
    worst_m = int(np.argmin(model_acc))
    oracle_any = np.any(pred_qm == correct[:, None], axis=1)
    return [{
        "reference": "best_single_model_oracle_known_after_labels",
        "acc": float(model_acc[best_m]),
        "model": SHORT_NAMES[best_m],
        "note": "not label-free; upper reference for unknown competence setting",
    }, {
        "reference": "worst_single_model",
        "acc": float(model_acc[worst_m]),
        "model": SHORT_NAMES[worst_m],
        "note": "shows ensemble heterogeneity",
    }, {
        "reference": "oracle_any_model_correct",
        "acc": float(np.mean(oracle_any)),
        "model": "oracle",
        "note": "not attainable; headroom for routing",
    }]


def make_paper_method_predictions(features_qm, p_qkm, correct):
    method_preds = {}
    method_scores = {}
    method_meta = {}

    for method in NAIVE_AGG_METHODS:
        preds, scores = aggregate_preds(p_qkm, method)
        method_preds[method] = preds
        method_scores[method] = scores
        method_meta[method] = {
            "family": "ordinal" if method in ORDINAL_AGG_METHODS else "probability",
            "base_method": method,
            "weighting": "none",
            "signal": "none",
            "temperature": "",
            "uses_probabilities": method in PROB_AGG_METHODS,
            "is_ours": False,
        }

    if REPORT_ENTROPY_WEIGHTING:
        ent_w = make_entropy_weighted_model_weights(p_qkm, temperature=ENTROPY_WEIGHT_TEMPERATURE)
        for method in NAIVE_AGG_METHODS:
            name = f"entropy_weighted_{method}"
            preds, scores = aggregate_preds(p_qkm, method, weights_qm=ent_w)
            method_preds[name] = preds
            method_scores[name] = scores
            method_meta[name] = {
                "family": "ordinal" if method in ORDINAL_AGG_METHODS else "probability",
                "base_method": method,
                "weighting": "entropy_softmax",
                "signal": "entropy",
                "temperature": float(ENTROPY_WEIGHT_TEMPERATURE),
                "uses_probabilities": True,
                "is_ours": False,
            }

    for signal in PAPER_STABILITY_FEATURES:
        if signal not in features_qm:
            raise KeyError(f"Missing required stability signal: {signal}")
        rel = zscore_per_question(np.asarray(features_qm[signal], dtype=np.float64))
        w = softmax_weights(rel, temperature=PAPER_STABILITY_TEMPERATURE)
        for method in NAIVE_AGG_METHODS:
            name = f"softmax_stability_{signal}_{method}"
            preds, scores = aggregate_preds(p_qkm, method, weights_qm=w)
            method_preds[name] = preds
            method_scores[name] = scores
            method_meta[name] = {
                "family": "ordinal" if method in ORDINAL_AGG_METHODS else "probability",
                "base_method": method,
                "weighting": "stability_softmax",
                "signal": signal,
                "temperature": float(PAPER_STABILITY_TEMPERATURE),
                "uses_probabilities": method in PROB_AGG_METHODS,
                "is_ours": True,
            }

    return method_preds, method_scores, method_meta


def paper_main_table_rows(method_preds, method_scores, method_meta, correct):
    rows = []
    base_acc = {}
    base_preds = {}
    for method in NAIVE_AGG_METHODS:
        preds = np.asarray(method_preds[method], dtype=np.int64)
        base_preds[method] = preds
        base_acc[method] = float(np.mean(preds == correct))

    for name, preds in method_preds.items():
        preds = np.asarray(preds, dtype=np.int64)
        meta = method_meta[name]
        base_method = meta["base_method"]
        acc = float(np.mean(preds == correct))
        base = base_preds[base_method]
        base_correct = base == correct
        method_correct = preds == correct
        wins = int(np.sum(method_correct & (~base_correct)))
        losses = int(np.sum((~method_correct) & base_correct))
        boot = bootstrap_delta_ci_p(preds, base, correct)
        scores = method_scores.get(name)
        opt_auc = option_level_auroc(scores, correct) if isinstance(scores, np.ndarray) and scores.ndim == 2 else float("nan")
        rows.append({
            "family": meta["family"],
            "method": name,
            "base_method": base_method,
            "weighting": meta["weighting"],
            "signal": meta["signal"],
            "temperature": meta["temperature"],
            "uses_probabilities": meta["uses_probabilities"],
            "is_ours": meta["is_ours"],
            "acc": acc,
            "matched_base_acc": base_acc[base_method],
            "delta_vs_matched_base": acc - base_acc[base_method],
            "wins": wins,
            "losses": losses,
            "net_wins": wins - losses,
            "mcnemar_p": mcnemar_exact_p(wins, losses),
            "bootstrap_ci95_low": boot.get("bootstrap_ci95_low", float("nan")),
            "bootstrap_ci95_high": boot.get("bootstrap_ci95_high", float("nan")),
            "bootstrap_p_two_sided": boot.get("bootstrap_p_two_sided", float("nan")),
            "option_auroc": opt_auc,
        })

    order_weighting = {"none": 0, "entropy_softmax": 1, "stability_softmax": 2}
    order_method = {m: i for i, m in enumerate(NAIVE_AGG_METHODS)}
    return sorted(rows, key=lambda r: (r["family"], order_method.get(r["base_method"], 99), order_weighting.get(r["weighting"], 99), r["signal"]))


def print_paper_summary_tables(single_rows, oracle_rows, paper_rows):
    print("\n" + "=" * 120)
    print("Paper-facing cyclic stability evaluation")
    print("=" * 120)
    print("Methods:")
    print("  Baselines: hard_majority, borda, mrr, irv, arithmetic_mean, geometric_mean")
    print("  Extra baseline: entropy-weighted versions of each aggregator")
    print("  Ours: softmax stability weighting with rbo_p85 and rank_rr_overlap_alpha2")
    print("  No grid search. Fixed main-paper temperature T=1.0.")

    print_table(single_rows, "Single-model accuracy: raw vs cyclic_mean control", headers=["setting", "model", "n", "acc", "correct", "incorrect"])
    print_table(oracle_rows, "Oracle/reference accuracies", headers=["reference", "acc", "model", "note"])

    baseline_rows = [r for r in paper_rows if r["weighting"] == "none"]
    entropy_rows = [r for r in paper_rows if r["weighting"] == "entropy_softmax"]
    ours_rows = [r for r in paper_rows if r["weighting"] == "stability_softmax"]

    print_table(
        baseline_rows,
        "Unweighted baselines",
        headers=["family", "base_method", "method", "acc", "option_auroc", "uses_probabilities"],
    )

    print_table(
        entropy_rows,
        "Entropy-weighted baselines",
        headers=[
            "family", "base_method", "method", "signal", "acc", "matched_base_acc",
            "delta_vs_matched_base", "wins", "losses", "mcnemar_p",
            "bootstrap_ci95_low", "bootstrap_ci95_high", "uses_probabilities",
        ],
    )

    print_table(
        ours_rows,
        "Ours: softmax cyclic stability weighting",
        headers=[
            "family", "base_method", "method", "signal", "acc", "matched_base_acc",
            "delta_vs_matched_base", "wins", "losses", "mcnemar_p",
            "bootstrap_ci95_low", "bootstrap_ci95_high", "uses_probabilities",
        ],
    )

    compact = []
    for base_method in NAIVE_AGG_METHODS:
        candidates = [r for r in ours_rows if r["base_method"] == base_method]
        if COMPACT_TABLE_FORCE_SIGNAL is not None:
            candidates = [r for r in candidates if r["signal"] == COMPACT_TABLE_FORCE_SIGNAL]
        if candidates:
            compact.append(max(candidates, key=lambda r: r["acc"]))

    title = "Compact paper table: rbo_p85 stability-weighted variant per aggregator"
    if COMPACT_TABLE_FORCE_SIGNAL is None:
        title = "Compact paper table: best stability-weighted variant per aggregator"

    print_table(
        compact,
        title,
        headers=[
            "family", "base_method", "signal", "matched_base_acc", "acc",
            "delta_vs_matched_base", "wins", "losses", "mcnemar_p",
            "bootstrap_ci95_low", "bootstrap_ci95_high", "uses_probabilities",
        ],
    )


# =============================================================================
# Main
# =============================================================================

def main():
    data = load_dataset_pack(N_SAMPLES, BENCHMARK, DATASET_SEED)
    correct = data.correct

    if not RUN_CYCLIC_SHIFT:
        raise RuntimeError("RUN_CYCLIC_SHIFT must be True for cyclic-stability analysis.")

    K = len(data.option_labels)
    cyclic_shifts = choose_cyclic_shifts(K=K, max_shifts=MAX_CYCLIC_SHIFTS, seed=CYCLIC_SHIFT_SEED)
    if 0 not in cyclic_shifts:
        raise RuntimeError("cyclic_shifts must include 0 so raw scores can be derived without a second model pass.")
    print(f"Cyclic shifts: {cyclic_shifts}")

    raw_gen, raw_cyclic = run_or_load_all_scores(data, cyclic_shifts)

    p_raw_qkm = get_mc_dists(raw_gen)
    p_cyc_qkms = get_cyclic_dists(raw_cyclic)
    p_cycmean_qkm = pool_cyclic_per_model(p_cyc_qkms, method="mean")
    Q, K, M = p_raw_qkm.shape

    print("Computing cyclic stability features ...")
    features = cyclic_stability_features(p_cyc_qkms, p_cycmean_qkm, shifts=cyclic_shifts)

    single_rows = single_model_rows(p_raw_qkm, p_cycmean_qkm, correct)
    oracle_rows = oracle_reference_rows(p_cycmean_qkm, correct)
    paper_method_preds, paper_method_scores, paper_method_meta = make_paper_method_predictions(features, p_cycmean_qkm, correct)
    paper_rows = paper_main_table_rows(paper_method_preds, paper_method_scores, paper_method_meta, correct)

    sensitivity_rows = sensitivity_rows_rbo_temperature(
        features_qm=features,
        p_qkm=p_cycmean_qkm,
        correct=correct,
    )
    sensitivity_csv_paths = save_sensitivity_csvs(
        rows=sensitivity_rows,
        out_dir=OUT_DIR,
        cache_tag=CACHE_TAG,
    )

    print(f"\nN={Q} | K={K} | M={M} | benchmark={data.benchmark_resolved} | seed={DATASET_SEED} | K_filter={data.k_filter}")
    print(f"Cache tag: {CACHE_TAG}")
    print(f"Cyclic cache: {CYCLIC_NPZ_FILE}")
    print(f"Stability signals: {PAPER_STABILITY_FEATURES}")
    print(f"Fixed stability temperature: T={PAPER_STABILITY_TEMPERATURE}")
    print(f"Sensitivity RBO p-grid: {RBO_PS}")
    print(f"Sensitivity T-grid: {SENSITIVITY_TEMPERATURES}")

    print_paper_summary_tables(single_rows=single_rows, oracle_rows=oracle_rows, paper_rows=paper_rows)
    print_sensitivity_table(sensitivity_rows)

    results: Dict[str, Any] = {
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
            "cyclic_shifts": cyclic_shifts,
            "cyclic_npz_file": CYCLIC_NPZ_FILE,
            "gen_npz_file": GEN_NPZ_FILE,
            "per_model_cache_dir": MODEL_CACHE_DIR,
            "ordinal_agg_methods": ORDINAL_AGG_METHODS,
            "prob_agg_methods": PROB_AGG_METHODS,
            "paper_stability_features": PAPER_STABILITY_FEATURES,
            "paper_stability_temperature": PAPER_STABILITY_TEMPERATURE,
            "entropy_weight_temperature": ENTROPY_WEIGHT_TEMPERATURE,
            "compact_table_force_signal": COMPACT_TABLE_FORCE_SIGNAL,
            "sensitivity_rbo_ps": RBO_PS,
            "sensitivity_temperatures": SENSITIVITY_TEMPERATURES,
            "sensitivity_methods": SENSITIVITY_METHODS,
            "framing": {
                "main_claim": "Cyclic rank stability is evaluated as a fixed, label-free, calibration-free soft weighting signal for matched aggregators.",
                "no_grid_search": True,
                "probability_caveat": "Arithmetic/geometric mean and entropy weighting use probability magnitudes and are calibration-sensitive references.",
                "sensitivity_note": "The p/T sweep is post-hoc and uses cached cyclic scores; it does not change model scoring or cache paths.",
            },
            "signal_definitions": {
                rbo_signal_name(float(p)): f"Mean pairwise finite-depth Rank-Biased Overlap across cyclic rankings with p={float(p):.2f}."
                for p in RBO_PS
            } | {
                "rank_rr_overlap_alpha2": "Mean pairwise cosine overlap of reciprocal-rank vectors 1/rank^2 across cyclic shifts.",
            },
        },
        "single_model_rows": single_rows,
        "oracle_reference_rows": oracle_rows,
        "paper_main_rows": paper_rows,
        "paper_method_meta": paper_method_meta,
        "sensitivity_rows": sensitivity_rows,
        "sensitivity_csv_paths": sensitivity_csv_paths,
        "feature_summary_stats": {
            name: {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "q10": float(np.quantile(arr, 0.10)),
                "q25": float(np.quantile(arr, 0.25)),
                "q50": float(np.quantile(arr, 0.50)),
                "q75": float(np.quantile(arr, 0.75)),
                "q90": float(np.quantile(arr, 0.90)),
            }
            for name, arr in features.items()
        },
    }

    with open(JSON_FILE, "w") as f:
        json.dump(serialise(results), f, indent=2)
    print(f"\nSaved compact JSON results: {JSON_FILE}")


if __name__ == "__main__":
    main()
