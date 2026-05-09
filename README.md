# Cyclic Rank Stability for Calibration-Free LLM Ensemble Aggregation

This repository contains anonymized code for reproducing the experiments from:

**Cyclic Rank Stability for Calibration-Free LLM Ensemble Aggregation**

The code implements cyclic option scoring, RBO-based cyclic rank-stability weighting, ordinal ensemble aggregation, ensemble-size ablations, heterogeneity analyses, signal-validity checks, and entropy-weighted comparators for multiple-choice LLM benchmarks.

---

## Overview

The method evaluates heterogeneous ensembles of instruction-tuned language models on multiple-choice benchmarks.

For each question, model, and cyclic answer-option rotation, the scoring script computes teacher-forced next-token scores for the answer labels. The scores are realigned to the original semantic answer options. A model is treated as more reliable on a question if its ranking of semantic answer options remains stable across cyclic rotations.

Reliability is measured using Rank-Biased Overlap (RBO). The resulting per-question, per-model stability values are converted into softmax weights and used with ordinal aggregation rules such as hard majority, Borda, MRR, and IRV.

The main comparison is between:

1. cyclic-averaged unweighted ordinal aggregation, and  
2. cyclic-averaged ordinal aggregation with RBO-based rank-stability weighting.

Thus, the experiments isolate the effect of reliability weighting beyond cyclic option-order averaging.

---

## Repository structure

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

| File | Description |
|---|---|
| `environment.yml` | Conda environment specification. |
| `.env.example` | Template for local environment variables. |
| `scripts/01_score_cyclic_stability.py` | Main cyclic scoring and aggregation script. |
| `scripts/02_ensemble_size_ablation.py` | Ensemble-size, heterogeneity, and signal-validity analyses. |
| `scripts/03_entropy_vs_rbo.py` | Entropy-vs-RBO comparison and calibration stress test. |

---

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate rbo-stability
```

The tested environment used:

```text
Python 3.11.15
torch 2.9.1+cu128
CUDA 12.8
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

---

## Environment variables

The scripts read local configuration from a `.env` file.

Create a local `.env` file from the provided template:

```bash
cp .env.example .env
```

Example `.env` content:

```bash
HF_TOKEN=hf_your_token_here
HF_CACHE_DIR=.cache/huggingface
OUTPUT_DIR=majority_regime_analysis_results
```

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face token. Required only for gated models. |
| `HF_CACHE_DIR` | Directory used for Hugging Face model and dataset caches. |
| `OUTPUT_DIR` | Directory used for generated score caches, JSON files, and CSV files. |

The `.env` file is intentionally excluded from version control.

---

## Benchmarks

The scripts support multiple-choice benchmarks loaded through Hugging Face `datasets`, including:

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

The benchmark is selected by editing the `BENCHMARK` variable near the top of each script.

---

## Models

Models are configured through the `MODEL_NAMES` list near the top of each script.

The reported experiments use Hugging Face model checkpoints. Some checkpoints may require gated access. For gated models, the user must have accepted the model terms on Hugging Face and must provide a valid `HF_TOKEN` in `.env`.

---

## Reproduction workflow

The scripts should be run in this order:

```bash
python scripts/01_score_cyclic_stability.py
python scripts/02_ensemble_size_ablation.py
python scripts/03_entropy_vs_rbo.py
```

`01_score_cyclic_stability.py` performs model inference and writes the cyclic score cache.

`02_ensemble_size_ablation.py` and `03_entropy_vs_rbo.py` reuse the cyclic score cache and do not rescore models.

---

# Script 1: Cyclic scoring and main aggregation analysis

```bash
python scripts/01_score_cyclic_stability.py
```

## Purpose

This script performs the main cyclic scoring and benchmark-level aggregation analysis.

## Procedure

The script:

1. loads the selected benchmark,
2. samples the configured number of examples using a fixed seed,
3. loads each model in `MODEL_NAMES`,
4. scores each model on all cyclic answer-option rotations,
5. realigns cyclic scores to the original semantic answer options,
6. caches per-model cyclic scores,
7. builds the full cyclic score tensor,
8. computes cyclic-mean option probabilities,
9. computes cyclic rank-stability features,
10. evaluates unweighted aggregation baselines,
11. evaluates entropy-weighted baselines,
12. evaluates RBO/rank-stability-weighted variants,
13. computes paired statistics,
14. saves JSON and CSV outputs.

## Main configuration variables

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

Main method settings:

```python
RUN_CYCLIC_SHIFT = True
MAX_CYCLIC_SHIFTS = None
CYCLIC_SHIFT_SEED = 123

RBO_PS = [0.85]
PAPER_STABILITY_FEATURES = ["rbo_p85", "rank_rr_overlap_alpha2"]
PAPER_STABILITY_TEMPERATURE = 1.0
ENTROPY_WEIGHT_TEMPERATURE = 1.0
```

## Outputs

Outputs are written to `OUTPUT_DIR`.

Typical outputs include:

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

## Main result file

The main JSON file is:

```text
paper_facing_stability_eval_*.json
```

It contains:

| Key | Description |
|---|---|
| `config` | Benchmark, model, cache, and method configuration. |
| `single_model_rows` | Raw and cyclic-mean single-model accuracies. |
| `oracle_reference_rows` | Oracle and reference accuracies. |
| `paper_main_rows` | Main aggregation results. |
| `paper_method_meta` | Metadata for each aggregation method. |
| `sensitivity_rows` | RBO persistence and temperature sensitivity results. |
| `feature_summary_stats` | Summary statistics for stability features. |

The most important result table is `paper_main_rows`.

Important fields:

| Field | Description |
|---|---|
| `method` | Aggregation method or weighted variant. |
| `base_method` | Matched unweighted baseline. |
| `weighting` | `none`, `entropy_softmax`, or `stability_softmax`. |
| `signal` | Reliability signal, e.g. `rbo_p85`. |
| `acc` | Accuracy of the method. |
| `matched_base_acc` | Accuracy of the matched baseline. |
| `delta_vs_matched_base` | Matched accuracy gain. |
| `wins` | Number of examples fixed by the weighted method. |
| `losses` | Number of examples broken by the weighted method. |
| `mcnemar_p` | Paired McNemar test p-value. |
| `bootstrap_ci95_low` | Lower bootstrap confidence interval for the matched delta. |
| `bootstrap_ci95_high` | Upper bootstrap confidence interval for the matched delta. |

---

# Script 2: Ensemble-size, heterogeneity, and signal-validity analysis

```bash
python scripts/02_ensemble_size_ablation.py
```

## Purpose

This script analyzes whether RBO-based rank-stability weighting transfers across sub-ensembles and whether the gains are related to model heterogeneity.

It also evaluates whether the RBO stability signal tracks model-question correctness.

## Dependency

This script requires the cyclic score cache produced by Script 1:

```text
cyclic_gen_scores_*.npz
```

The following settings must match the scoring run:

```python
MODEL_NAMES
BENCHMARK
N_SAMPLES
DATASET_SEED
DATASET_K_FILTER
MAX_CYCLIC_SHIFTS
CYCLIC_SHIFT_SEED
```

If these values do not match, the script raises a cache mismatch error.

## Procedure

The script:

1. loads the same benchmark selection as Script 1,
2. loads the matching cyclic score cache,
3. computes cyclic-mean probabilities,
4. precomputes rankings and predictions,
5. computes RBO `p = 0.85` stability features,
6. computes signal-validity statistics,
7. enumerates sub-ensembles of each ensemble size,
8. evaluates ordinal aggregation rules,
9. compares RBO-weighted variants against matched unweighted baselines,
10. computes heterogeneity metrics,
11. computes correlations between heterogeneity and gains,
12. writes row-level and summary result files.

## Main configuration variables

```python
MODEL_NAMES = [...]
BENCHMARK = "mmlu"
N_SAMPLES = 14042
DATASET_SEED = 42
DATASET_K_FILTER = "modal"

MAX_SUBENSEMBLES_PER_SIZE = None
SUBENSEMBLE_SAMPLE_SEED = 12345
COMPUTE_ROW_MCNEMAR = False

HETEROGENEITY_N_BINS = 3
HETEROGENEITY_PRIMARY_METHOD = "mrr"
```

## Outputs

Typical outputs include:

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

## Result files

### `ensemble_size_ablation_rows_*.csv`

This file contains one row per:

```text
ensemble size × sub-ensemble × aggregation method
```

Important columns:

| Column | Description |
|---|---|
| `ensemble_size` | Number of models in the sub-ensemble. |
| `subset_indices` | Indices of models in the sub-ensemble. |
| `subset_models` | Names of models in the sub-ensemble. |
| `base_method` | Unweighted ordinal aggregation rule. |
| `weighted_method` | RBO-weighted variant. |
| `base_acc` | Accuracy of the unweighted baseline. |
| `weighted_acc` | Accuracy after RBO weighting. |
| `delta` | Matched accuracy gain. |
| `wins` | Examples fixed by RBO weighting. |
| `losses` | Examples broken by RBO weighting. |
| `single_acc_range` | Competence spread within the sub-ensemble. |
| `single_acc_std` | Standard deviation of single-model accuracies. |

### `ensemble_size_ablation_summary_*.csv`

This file aggregates row-level results by ensemble size and base method.

Important columns:

| Column | Description |
|---|---|
| `ensemble_size` | Number of models. |
| `base_method` | Aggregation rule. |
| `n_subensembles` | Number of evaluated sub-ensembles. |
| `base_acc_mean` | Mean unweighted accuracy. |
| `weighted_acc_mean` | Mean RBO-weighted accuracy. |
| `delta_mean` | Mean matched gain. |
| `delta_q05`, `delta_q50`, `delta_q95` | Gain quantiles. |
| `frac_delta_positive` | Fraction of sub-ensembles improved by weighting. |
| `corr_single_acc_range_delta` | Correlation between heterogeneity and gain. |

### `signal_validity_by_model_*.csv`

This file reports whether RBO stability tracks correctness within each model.

Important columns:

| Column | Description |
|---|---|
| `model` | Model name. |
| `accuracy` | Cyclic-mean single-model accuracy. |
| `phi_mean` | Mean RBO stability. |
| `phi_correct_mean` | Mean stability for correct predictions. |
| `phi_wrong_mean` | Mean stability for wrong predictions. |
| `phi_correct_minus_wrong` | Difference between correct and wrong cases. |
| `within_model_r` | Correlation between stability and correctness. |
| `within_model_p` | p-value for the within-model correlation. |

---

# Script 3: Entropy-vs-RBO comparison and calibration stress test

```bash
python scripts/03_entropy_vs_rbo.py
```

## Purpose

This script compares RBO-based rank-stability weighting against entropy-based confidence weighting.

It also runs a controlled stress test where the entropy signal of the weakest model is artificially sharpened while the actual aggregation probabilities remain unchanged.

## Dependency

This script requires the cyclic score cache produced by Script 1:

```text
cyclic_gen_scores_*.npz
```

The following settings must match the scoring run:

```python
MODEL_NAMES
BENCHMARK
N_SAMPLES
DATASET_SEED
DATASET_K_FILTER
MAX_CYCLIC_SHIFTS
CYCLIC_SHIFT_SEED
```

## Procedure

The script:

1. loads the same benchmark selection as Script 1,
2. loads the matching cyclic score cache,
3. computes cyclic-mean probabilities,
4. computes RBO stability features,
5. computes entropy-based confidence weights,
6. computes RBO-based rank-stability weights,
7. compares entropy and RBO weighting across aggregation rules,
8. identifies the weakest cyclic-mean model,
9. sharpens only that model's entropy signal,
10. keeps the actual aggregation probabilities fixed,
11. recomputes entropy weights under different sharpening temperatures,
12. reports how entropy weighting changes under spurious overconfidence.

## Main configuration variables

```python
RBO_P = 0.85
STABILITY_TEMPERATURE = 1.0
ENTROPY_TEMPERATURE = 1.0
```

Stress-test temperatures:

```python
TOY_ENTROPY_SIGNAL_TAU_GRID = [
    1.0, 0.75, 0.5, 0.333333, 0.25, 0.2,
    0.166667, 0.125, 0.1, 0.05, 0.02, 0.01
]
```

## Outputs

This script prints diagnostic tables to stdout:

| Table | Description |
|---|---|
| `Direct paired comparison: entropy weighting vs RBO-rank weighting` | Direct matched comparison between entropy and RBO weighting. |
| `Toy diagnostics: sanity checks for isolated perturbation` | Checks whether top-1 predictions and rankings changed under the entropy-signal perturbation. |
| `Toy overconfidence sweep: entropy vs RBO` | Accuracy comparison across sharpening temperatures. |
| `Toy sweep compact summary across ordinal aggregators` | Compact mean ordinal comparison across temperatures. |

Important fields:

| Field | Description |
|---|---|
| `base_acc` | Accuracy of the unweighted cyclic-mean aggregator. |
| `entropy_acc` | Accuracy with entropy weighting. |
| `rbo_acc` | Accuracy with RBO-rank weighting. |
| `entropy_minus_rbo` | Difference between entropy and RBO weighting. |
| `weak_model_entropy_weight` | Mean entropy weight assigned to the weakest model. |
| `weak_model_rbo_weight` | Mean RBO weight assigned to the weakest model. |
| `top1_changed_weak_model` | Fraction of changed top-1 predictions for the perturbed model. |
| `full_ranking_changed_weak_model` | Fraction of changed full rankings for the perturbed model. |

---

## Benchmark configurations

The benchmark configuration is controlled at the top of each script.

Examples:

### MMLU

```python
BENCHMARK = "mmlu"
N_SAMPLES = 14042
DATASET_SEED = 42
DATASET_K_FILTER = "modal"
```

### MMLU-Pro

```python
BENCHMARK = "mmlu_pro"
N_SAMPLES = 9981
DATASET_SEED = 42
DATASET_K_FILTER = "modal"
```

### MedQA

```python
BENCHMARK = "medqa"
N_SAMPLES = 1274
DATASET_SEED = 42
DATASET_K_FILTER = "modal"
```

### ARC-Challenge

```python
BENCHMARK = "arc_challenge"
N_SAMPLES = 1165
DATASET_SEED = 42
DATASET_K_FILTER = "modal"
```

### GPQA-Extended

```python
BENCHMARK = "gpqa_extended"
N_SAMPLES = 546
DATASET_SEED = 42
DATASET_K_FILTER = "modal"
```

After changing the benchmark or model list, rerun Script 1 before running Scripts 2 or 3.

---

## Cache naming and matching

Cache names are derived from:

```python
CACHE_TAG = f"{BENCHMARK}_K{DATASET_K_FILTER}_N{N_SAMPLES}_seed{DATASET_SEED}_M{len(MODEL_NAMES)}"
```

The cyclic cache additionally depends on:

```python
MAX_CYCLIC_SHIFTS
CYCLIC_SHIFT_SEED
```

Downstream scripts require the cache to match the current configuration exactly.

If a cache mismatch occurs, verify:

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

## Compute requirements

Cyclic scoring is the dominant computational cost.

For:

```text
Q = number of questions
M = number of models
K = number of answer options
```

the approximate scoring cost is:

```text
Q × M × K
```

teacher-forced forward passes.

The scripts cache intermediate cyclic scores so that aggregation analyses, sensitivity checks, heterogeneity analyses, bootstrap intervals, and statistical tests can be recomputed without rerunning model inference.

Runtime and memory depend on:

- model size,
- number of models,
- benchmark size,
- number of answer options,
- sequence length,
- GPU memory,
- scoring batch size.

If GPU memory is insufficient, reduce:

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

## Troubleshooting

### Gated Hugging Face model

If a gated model cannot be loaded, verify that:

1. the model terms have been accepted on Hugging Face,
2. a valid token is present in `.env`,
3. the token has access to the requested model.

### CUDA out of memory

Reduce `SCORE_BATCH` or `MAX_LENGTH`.

### Cache mismatch

Ensure that the downstream scripts use the same benchmark, model list, sample size, seed, and cyclic-shift settings as the scoring run.

### Missing package

Check the environment:

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

## Fast smoke test

For a quick functionality check, reduce the configuration in Script 1:

```python
N_SAMPLES = 10
MODEL_NAMES = [
    "google/gemma-3-270m-it",
]
```

Then run:

```bash
python scripts/01_score_cyclic_stability.py
```

This does not reproduce the reported results, but it checks the environment, dataset loading, model loading, scoring, and output writing.

---

## Reproducibility notes

The scripts use fixed seeds for dataset sampling and cyclic-shift selection. Reproducibility depends on matching:

- model checkpoints,
- dataset versions,
- Python package versions,
- benchmark configuration,
- model list,
- sample size,
- random seeds,
- cache settings.

The provided `environment.yml` records the tested software environment.

---

## License and citation

This repository is provided as anonymized supplementary code for review.

Citation and license information will be added after de-anonymization.
