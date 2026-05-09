# Cyclic Rank Stability for Calibration-Free LLM Ensemble Aggregation

This repository contains the anonymized experimental code for the paper:

**Cyclic Rank Stability for Calibration-Free LLM Ensemble Aggregation**

The code reproduces the main computational pipeline for cyclic option scoring, RBO-based cyclic rank-stability weighting, ordinal ensemble aggregation, ensemble-size ablations, heterogeneity analyses, and entropy-weighted comparators.

The repository is intended for anonymous review. It contains only code and setup files, not local caches, Hugging Face tokens, generated `.npz` files, or machine-specific paths.

---

## 1. Method overview

The experiments evaluate heterogeneous ensembles of instruction-tuned LLMs on multiple-choice benchmarks.

For each question, model, and cyclic answer-option rotation, the scoring script computes teacher-forced next-token scores for all answer labels. The scores are then realigned to the original semantic answer options.

The core reliability signal is **cyclic rank stability**:

- A model is considered more reliable on a question if its ranking of semantic answer options remains stable across cyclic answer-label rotations.
- Rank stability is measured with Rank-Biased Overlap (RBO).
- The fixed main-paper setting uses `p = 0.85` for RBO and softmax temperature `T = 1.0` for reliability weights.
- The weighted aggregators are compared against matched cyclic-averaged unweighted baselines.

The main point of the comparison is to isolate the effect of reliability weighting, not simply the effect of option-order debiasing. All ordinal baselines already use cyclic-mean scores.

---

## 2. Repository structure

```text
.
├── README.md
├── environment.yml
├── .env.example
├── .gitignore
└── scripts/
    ├── 01_score_cyclic_stability.py
    ├── 02_ensemble_size_ablation.py
    └── 03_entropy_vs_rbo.py
```

### Files

| File | Purpose |
|---|---|
| `README.md` | This documentation file. |
| `environment.yml` | Conda environment specification. |
| `.env.example` | Template for local environment variables such as Hugging Face token, cache path, and output path. |
| `.gitignore` | Prevents committing secrets, local caches, generated outputs, and Python cache files. |
| `scripts/01_score_cyclic_stability.py` | Main scoring and benchmark-level aggregation script. |
| `scripts/02_ensemble_size_ablation.py` | Sub-ensemble, ensemble-size, heterogeneity, and signal-validity analysis. |
| `scripts/03_entropy_vs_rbo.py` | Entropy-vs-RBO comparison and calibration stress test. |

---

## 3. Environment setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate rbo-stability
```

The environment used for the reported code path was tested with:

```text
Python 3.11.15
torch 2.9.1+cu128
CUDA 12.8
NVIDIA H100 80GB HBM3
numpy 2.4.3
scipy 1.17.1
transformers 5.8.0
datasets 4.8.4
huggingface_hub 1.7.2
accelerate 1.13.0
sentencepiece 0.2.1
safetensors 0.7.0
python-dotenv 1.2.2
```

A matching `environment.yml` is:

```yaml
name: rbo-stability
channels:
  - pytorch
  - nvidia
  - conda-forge

dependencies:
  - python=3.11
  - pip
  - numpy=2.4.3
  - scipy=1.17.1
  - pip:
      - --extra-index-url https://download.pytorch.org/whl/cu128
      - torch==2.9.1+cu128
      - transformers==5.8.0
      - datasets==4.8.4
      - huggingface_hub==1.7.2
      - accelerate==1.13.0
      - sentencepiece==0.2.1
      - safetensors==0.7.0
      - python-dotenv==1.2.2
```

If exact versions are unavailable on a different cluster, use the closest compatible versions and record the tested environment.

---

## 4. Local `.env` configuration

Copy the example file:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
HF_TOKEN=hf_your_token_here
HF_CACHE_DIR=.cache/huggingface
OUTPUT_DIR=majority_regime_analysis_results
```

### Variables

| Variable | Required? | Description |
|---|---:|---|
| `HF_TOKEN` | Optional, but required for gated models | Hugging Face access token. Needed for models such as Llama if access is gated. |
| `HF_CACHE_DIR` | Optional | Local Hugging Face cache directory. Defaults to `.cache/huggingface`. |
| `OUTPUT_DIR` | Optional | Directory where caches, JSON files, and CSV outputs are written. Defaults to `majority_regime_analysis_results`. |

The `.env` file must not be committed.

Only `.env.example` should be uploaded to GitHub.

---

## 5. Anonymity and files not to commit

Do not commit:

```text
.env
.cache/
hf_cache/
majority_regime_analysis_results/
*.npz
local cluster paths
Hugging Face tokens
user names
institution names
```

Before uploading or committing, search the scripts for local identifiers:

```bash
grep -R "/vol/" .
grep -R "HF_TOKEN=hf_" .
grep -R "ak95ecuh" .
```

The code should contain only environment-variable based token handling, e.g.:

```python
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
```

---

## 6. Data and models

The scripts load benchmark datasets through Hugging Face `datasets`.

Supported benchmark aliases in the scripts include:

```text
mmlu
mmlu_pro
arc_easy
arc_challenge
race
race_high
race_middle
gpqa_diamond
gpqa_main
gpqa_extended
gpqa_experts
medqa
```

The model checkpoints are loaded from Hugging Face using `transformers`.

Some checkpoints require:

1. A Hugging Face account.
2. Acceptance of the model terms on the model page.
3. A valid `HF_TOKEN` in `.env`.

---

## 7. Typical execution order

Run the scripts in this order:

```bash
python scripts/01_score_cyclic_stability.py
python scripts/02_ensemble_size_ablation.py
python scripts/03_entropy_vs_rbo.py
```

Script 1 performs model inference and creates the cyclic score cache.

Scripts 2 and 3 are downstream analyses. They expect the cyclic cache generated by Script 1 and do not rescore models.

---

# 8. Script 1: Cyclic scoring and main aggregation analysis

```bash
python scripts/01_score_cyclic_stability.py
```

## Purpose

This is the main experimental script. It performs cyclic option scoring, computes cyclic rank-stability features, evaluates aggregation rules, and writes paper-facing result files.

## What it does

The script:

1. Loads the selected benchmark.
2. Applies the configured answer-option count filter.
3. Samples `N_SAMPLES` examples using `DATASET_SEED`.
4. Loads each model listed in `MODEL_NAMES`.
5. Scores each model on all cyclic answer-option rotations.
6. Realigns shifted option scores back to original semantic option identities.
7. Caches per-model cyclic scores.
8. Builds the full cyclic score tensor.
9. Computes cyclic-mean probabilities.
10. Computes rank-stability features, including RBO-based stability.
11. Evaluates single-model raw vs cyclic-mean accuracy.
12. Evaluates unweighted aggregation baselines.
13. Evaluates entropy-weighted baselines.
14. Evaluates RBO/rank-stability-weighted variants.
15. Computes paired tests, bootstrap intervals, and option-level AUROC.
16. Saves JSON and CSV result files.
17. Prints paper-facing summary tables.

## Main configuration block

At the top of the script, edit:

```python
MODEL_NAMES = [...]
BENCHMARK = "mmlu"
N_SAMPLES = 14042
DATASET_SEED = 42
DATASET_K_FILTER = "modal"
USE_CACHE = True
SCORE_BATCH = 64
MAX_LENGTH = 2048
```

Important method settings:

```python
RUN_CYCLIC_SHIFT = True
MAX_CYCLIC_SHIFTS = None
CYCLIC_SHIFT_SEED = 123

RBO_PS = [0.85]
PAPER_STABILITY_FEATURES = ["rbo_p85", "rank_rr_overlap_alpha2"]
PAPER_STABILITY_TEMPERATURE = 1.0
ENTROPY_WEIGHT_TEMPERATURE = 1.0
```

## Important outputs

Outputs are written to `OUTPUT_DIR`, for example:

```text
majority_regime_analysis_results/
├── per_model_cache/
├── cyclic_gen_scores_*.npz
├── gen_scores_*.npz
├── paper_facing_stability_eval_*.json
├── rbo_sensitivity_all_*.csv
├── rbo_sensitivity_hard_majority_*.csv
├── rbo_sensitivity_borda_*.csv
├── rbo_sensitivity_mrr_*.csv
└── rbo_sensitivity_irv_*.csv
```

## Cache behavior

The script writes two cache types:

### Per-model cache

```text
majority_regime_analysis_results/per_model_cache/
```

This stores cyclic scores per model and can be reused if a run is interrupted.

### Full cyclic run cache

```text
cyclic_gen_scores_*.npz
```

This stores the full tensor of cyclic scores for all models.

If `USE_CACHE = True`, existing compatible caches are reused.

If model names, benchmark, sample size, option count, prompt template, or cyclic-shift settings differ, the script will create or require a different cache.

## How to evaluate the results

The main printed tables are:

- `Single-model accuracy: raw vs cyclic_mean control`
- `Oracle/reference accuracies`
- `Unweighted baselines`
- `Entropy-weighted baselines`
- `Ours: softmax cyclic stability weighting`
- `Compact paper table`
- `RBO-p / temperature sensitivity summary`

The main JSON file is:

```text
paper_facing_stability_eval_*.json
```

It contains:

```text
config
single_model_rows
oracle_reference_rows
paper_main_rows
paper_method_meta
sensitivity_rows
sensitivity_csv_paths
feature_summary_stats
```

The most important table for benchmark-level comparison is `paper_main_rows`. Each row includes:

| Field | Meaning |
|---|---|
| `method` | Aggregation method or weighted variant. |
| `base_method` | Matched unweighted base rule. |
| `weighting` | `none`, `entropy_softmax`, or `stability_softmax`. |
| `signal` | Reliability signal, e.g. `rbo_p85`. |
| `acc` | Accuracy of the method. |
| `matched_base_acc` | Accuracy of the matched unweighted baseline. |
| `delta_vs_matched_base` | Matched improvement over the baseline. |
| `wins` | Number of examples fixed by the weighted method. |
| `losses` | Number of examples broken by the weighted method. |
| `mcnemar_p` | Paired McNemar test p-value. |
| `bootstrap_ci95_low` | Lower bootstrap CI for matched delta. |
| `bootstrap_ci95_high` | Upper bootstrap CI for matched delta. |

---

# 9. Script 2: Ensemble-size, heterogeneity, and signal-validity analysis

```bash
python scripts/02_ensemble_size_ablation.py
```

## Purpose

This script evaluates whether RBO-based rank-stability weighting works only for the full ensemble or also transfers to smaller sub-ensembles. It also analyzes whether gains are stronger for heterogeneous sub-ensembles and whether the stability signal tracks model-question correctness.

## Dependency on Script 1

This script does not run model inference.

It requires the full cyclic cache produced by Script 1:

```text
cyclic_gen_scores_*.npz
```

The following settings must match the Script 1 run:

```python
MODEL_NAMES
BENCHMARK
N_SAMPLES
DATASET_SEED
DATASET_K_FILTER
MAX_CYCLIC_SHIFTS
CYCLIC_SHIFT_SEED
```

If they do not match, the script raises a cache mismatch error.

## What it does

The script:

1. Loads the same benchmark selection.
2. Loads the matching cyclic score cache.
3. Computes cyclic-mean probabilities.
4. Precomputes rankings and predictions.
5. Computes RBO `p = 0.85` stability features.
6. Computes signal-validity diagnostics.
7. Enumerates all sub-ensembles of each ensemble size.
8. Evaluates ordinal aggregation rules:
   - hard majority
   - Borda
   - MRR
   - IRV
9. Compares RBO-weighted variants against matched unweighted baselines.
10. Computes heterogeneity measures:
   - single-model accuracy range
   - single-model accuracy standard deviation
   - mean RBO range
   - mean RBO standard deviation
11. Computes correlations between heterogeneity and gains.
12. Bins sub-ensembles into low/medium/high heterogeneity groups.
13. Saves row-level CSVs, summary CSVs, and JSON files.

## Main configuration block

Edit near the top:

```python
MODEL_NAMES = [...]
BENCHMARK = "mmlu"
N_SAMPLES = 14042
DATASET_SEED = 42
DATASET_K_FILTER = "modal"
```

Sub-ensemble settings:

```python
MAX_SUBENSEMBLES_PER_SIZE = None
SUBENSEMBLE_SAMPLE_SEED = 12345
COMPUTE_ROW_MCNEMAR = False
```

Heterogeneity settings:

```python
HETEROGENEITY_N_BINS = 3
HETEROGENEITY_PRIMARY_METHOD = "mrr"
```

## Important outputs

```text
majority_regime_analysis_results/
├── ensemble_size_ablation_*.json
├── ensemble_size_ablation_rows_*.csv
├── ensemble_size_ablation_summary_*.csv
├── ensemble_size_heterogeneity_corr_*.csv
├── ensemble_size_heterogeneity_bins_range_*.csv
├── ensemble_size_heterogeneity_bins_std_*.csv
├── signal_validity_model_question_*.csv
├── signal_validity_by_model_*.csv
└── signal_validity_model_question_rows_*.csv
```

## How to evaluate the results

### `ensemble_size_ablation_rows_*.csv`

This is the row-level result file. Each row corresponds to:

```text
one ensemble size
one sub-ensemble
one aggregation rule
```

Important columns:

| Column | Meaning |
|---|---|
| `ensemble_size` | Number of models in the sub-ensemble. |
| `subset_indices` | Model indices used. |
| `subset_models` | Model names used. |
| `base_method` | Unweighted ordinal aggregator. |
| `weighted_method` | RBO-weighted variant. |
| `base_acc` | Accuracy of unweighted baseline. |
| `weighted_acc` | Accuracy after RBO weighting. |
| `delta` | Accuracy gain. |
| `wins` | Examples fixed by RBO weighting. |
| `losses` | Examples broken by RBO weighting. |
| `single_acc_range` | Difference between strongest and weakest model in the sub-ensemble. |
| `single_acc_std` | Standard deviation of single-model accuracies. |

### `ensemble_size_ablation_summary_*.csv`

This averages results over all sub-ensembles of the same size and aggregation method.

Important columns:

| Column | Meaning |
|---|---|
| `ensemble_size` | Number of models. |
| `base_method` | Aggregation rule. |
| `n_subensembles` | Number of evaluated sub-ensembles. |
| `base_acc_mean` | Mean unweighted accuracy. |
| `weighted_acc_mean` | Mean RBO-weighted accuracy. |
| `delta_mean` | Mean gain. |
| `delta_q05`, `delta_q50`, `delta_q95` | Gain quantiles. |
| `frac_delta_positive` | Fraction of sub-ensembles improved by RBO weighting. |
| `corr_single_acc_range_delta` | Correlation between heterogeneity and gain. |

### `signal_validity_by_model_*.csv`

This evaluates whether the RBO signal tracks correctness within each model.

Important columns:

| Column | Meaning |
|---|---|
| `model` | Model name. |
| `accuracy` | Cyclic-mean single-model accuracy. |
| `phi_mean` | Mean RBO stability. |
| `phi_correct_mean` | Mean RBO stability on correct predictions. |
| `phi_wrong_mean` | Mean RBO stability on wrong predictions. |
| `phi_correct_minus_wrong` | Difference between correct and wrong cases. |
| `within_model_r` | Point-biserial correlation between stability and correctness. |
| `within_model_p` | p-value for the within-model correlation. |

Use this script to reproduce the ensemble-size transfer analysis, heterogeneity-bin analysis, and signal-validity analysis.

---

# 10. Script 3: Entropy vs RBO comparison and calibration stress test

```bash
python scripts/03_entropy_vs_rbo.py
```

## Purpose

This script compares RBO-based rank-stability weighting against entropy-based confidence weighting. It also runs a controlled stress test in which the entropy signal of the weakest model is artificially sharpened while the actual aggregation probabilities are kept fixed.

## Dependency on Script 1

This script also requires the cyclic score cache from Script 1.

The following settings must match:

```python
MODEL_NAMES
BENCHMARK
N_SAMPLES
DATASET_SEED
DATASET_K_FILTER
MAX_CYCLIC_SHIFTS
CYCLIC_SHIFT_SEED
```

## What it does

The script:

1. Loads the same benchmark selection.
2. Loads the matching cyclic score cache.
3. Computes cyclic-mean probabilities.
4. Computes RBO stability features.
5. Computes entropy-based weights.
6. Computes RBO-based weights.
7. Compares entropy weighting and RBO weighting for:
   - hard majority
   - Borda
   - MRR
   - IRV
   - arithmetic mean
   - geometric mean
8. Identifies the weakest cyclic-mean model.
9. Artificially sharpens only that model's entropy signal.
10. Leaves the actual aggregation probabilities unchanged.
11. Recomputes entropy weights under different sharpening temperatures.
12. Shows how entropy weighting can degrade under spurious overconfidence.
13. Shows that RBO weighting remains unchanged under this probability-scale perturbation.

## Main configuration block

```python
RBO_P = 0.85
STABILITY_TEMPERATURE = 1.0
ENTROPY_TEMPERATURE = 1.0
```

Toy stress-test temperatures:

```python
TOY_ENTROPY_SIGNAL_TAU_GRID = [
    1.0, 0.75, 0.5, 0.333333, 0.25, 0.2,
    0.166667, 0.125, 0.1, 0.05, 0.02, 0.01
]
```

## Output

This script primarily prints diagnostic tables to stdout:

- `Direct paired comparison: entropy weighting vs RBO-rank weighting`
- `Toy diagnostics: sanity checks for isolated perturbation`
- `Toy overconfidence sweep: entropy vs RBO`
- `Toy sweep compact summary across ordinal aggregators`

The most important fields are:

| Field | Meaning |
|---|---|
| `base_acc` | Accuracy of unweighted cyclic-mean aggregator. |
| `entropy_acc` | Accuracy with entropy weighting. |
| `rbo_acc` | Accuracy with RBO-rank weighting. |
| `entropy_minus_rbo` | Difference between entropy and RBO weighting. |
| `weak_model_entropy_weight` | Mean weight assigned to the weakest model by entropy weighting. |
| `weak_model_rbo_weight` | Mean weight assigned to the weakest model by RBO weighting. |
| `top1_changed_weak_model` | Sanity check that top-1 predictions remain unchanged. |
| `full_ranking_changed_weak_model` | Sanity check for ranking changes. |

The stress test is diagnostic. It is designed to isolate calibration sensitivity of entropy weighting rather than to create a new benchmark result.

---

## 11. Reproducing a benchmark

To reproduce a benchmark, edit the configuration block at the top of each script.

Example for MMLU:

```python
BENCHMARK = "mmlu"
N_SAMPLES = 14042
DATASET_SEED = 42
DATASET_K_FILTER = "modal"
```

Then run:

```bash
python scripts/01_score_cyclic_stability.py
python scripts/02_ensemble_size_ablation.py
python scripts/03_entropy_vs_rbo.py
```

For MMLU-Pro, use:

```python
BENCHMARK = "mmlu_pro"
N_SAMPLES = 9981
DATASET_K_FILTER = "modal"
```

For MedQA, use:

```python
BENCHMARK = "medqa"
N_SAMPLES = 1274
DATASET_K_FILTER = "modal"
```

For ARC-Challenge, use:

```python
BENCHMARK = "arc_challenge"
N_SAMPLES = 1165
DATASET_K_FILTER = "modal"
```

For GPQA-Extended, use:

```python
BENCHMARK = "gpqa_extended"
N_SAMPLES = 546
DATASET_K_FILTER = "modal"
```

Always run Script 1 first after changing the benchmark or model list.

---

## 12. Matching cache tags

Cache file names are derived from:

```python
CACHE_TAG = f"{BENCHMARK}_K{DATASET_K_FILTER}_N{N_SAMPLES}_seed{DATASET_SEED}_M{len(MODEL_NAMES)}"
```

The cyclic cache additionally includes:

```python
MAX_CYCLIC_SHIFTS
CYCLIC_SHIFT_SEED
```

If downstream scripts cannot find or load a cache, check that the following values match the scoring run:

```python
MODEL_NAMES
BENCHMARK
N_SAMPLES
DATASET_SEED
DATASET_K_FILTER
MAX_CYCLIC_SHIFTS
CYCLIC_SHIFT_SEED
```

---

## 13. Compute notes

Cyclic scoring is the dominant cost.

For:

```text
Q = number of questions
M = number of models
K = number of answer options
```

the scoring cost scales approximately as:

```text
Q × M × K
```

teacher-forced forward passes.

The scripts use caching so that expensive model scoring is performed once. Aggregation analyses, sensitivity checks, heterogeneity analyses, bootstrap intervals, and McNemar tests can then be recomputed from cached scores.

Memory and runtime depend heavily on:

- model size,
- GPU memory,
- batch size,
- maximum sequence length,
- benchmark size,
- number of answer options.

If you encounter GPU out-of-memory errors, reduce:

```python
SCORE_BATCH = 64
MAX_LENGTH = 2048
```

For example:

```python
SCORE_BATCH = 16
MAX_LENGTH = 1536
```

---

## 14. Troubleshooting

### Hugging Face gated model error

If a gated model cannot be loaded:

1. Accept the model terms on Hugging Face.
2. Add a valid token to `.env`:

```bash
HF_TOKEN=hf_your_token_here
```

3. Re-run the script.

### CUDA out-of-memory

Reduce:

```python
SCORE_BATCH
MAX_LENGTH
```

or evaluate fewer/lower-memory models.

### Cache mismatch

If Script 2 or Script 3 raises a cache mismatch, ensure that their configuration exactly matches the Script 1 run.

Most common causes:

- changed `MODEL_NAMES`,
- changed benchmark,
- changed `N_SAMPLES`,
- changed `DATASET_K_FILTER`,
- changed cyclic-shift settings,
- deleted or moved output directory.

### Missing Python package

Activate the environment:

```bash
conda activate rbo-stability
```

Then check imports:

```bash
python - <<'PY'
import numpy
import scipy
import torch
import transformers
import datasets
import huggingface_hub
import accelerate
import sentencepiece
import safetensors
from dotenv import load_dotenv

print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("torch cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("accelerate", accelerate.__version__)
print("All imports OK")
PY
```

### Syntax check

Before running long jobs:

```bash
python -m py_compile scripts/01_score_cyclic_stability.py
python -m py_compile scripts/02_ensemble_size_ablation.py
python -m py_compile scripts/03_entropy_vs_rbo.py
```

---

## 15. Expected workflow for reviewers

A typical reviewer workflow is:

```bash
conda env create -f environment.yml
conda activate rbo-stability

cp .env.example .env
# edit .env if gated models are used

python scripts/01_score_cyclic_stability.py
python scripts/02_ensemble_size_ablation.py
python scripts/03_entropy_vs_rbo.py
```

For faster smoke testing, reviewers can reduce:

```python
N_SAMPLES
MODEL_NAMES
SCORE_BATCH
```

However, changing these values changes the cache tag and will not reproduce the reported full-scale results.

---

## 16. License and citation

This repository is released for anonymous review.

Citation information and license details will be added after de-anonymization.