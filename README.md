# BasisCF — Temporal Basis-Guided Counterfactual Explanations for Multivariate Time Series

Reference implementation for the paper **"BasisCF: A Temporal Basis-Guided Counterfactual
Explanations for Multivariate Time Series"** (`paper/counterf_basis.pdf`).

Instead of optimising a counterfactual perturbation directly over raw time steps — which tends to
produce jittery, physically implausible edits — BasisCF parameterises the perturbation in a
**low-dimensional temporal basis space** (B-spline, Fourier, RBF, Polynomial, Wavelet). Temporal
coherence is therefore built into the parameterisation rather than added as a post-hoc penalty. A
**Determinantal Point Process (DPP)** term in the coefficient space yields diverse yet valid
counterfactual trajectories.

Given a fixed black-box model $f$ and a query sequence $\mathbf{X} \in \mathbb{R}^{T \times D}$:

$$
\mathbf{X}_{cf} = \mathbf{X} + \mathbf{\Phi}\mathbf{W}, \qquad
\mathbf{\Phi} \in \mathbb{R}^{T \times K},\; \mathbf{W} \in \mathbb{R}^{K \times D},\; K \ll T
$$

Only the $K \times D$ coefficients $\mathbf{W}$ are optimised, under a joint objective over
**validity**, **proximity**, **group sparsity**, **smoothness**, and **diversity**.

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Training the black-box models](#training-the-black-box-models)
- [Generating counterfactuals](#generating-counterfactuals)
- [Reproducing the paper experiments](#reproducing-the-paper-experiments)
- [Baselines](#baselines)
- [Evaluation metrics](#evaluation-metrics)
- [Citation](#citation)

---

## Repository layout

```
├── src/
│   ├── counterfactuals/
│   │   ├── core.py          # BasisGenerator, TSFeatureSchema, TargetSpec,
│   │   │                    # GeneratorConfig, LossWeights
│   │   ├── basis.py         # Polynomial / Fourier / RBF / BSpline / Wavelet bases
│   │   ├── losses.py        # validity, proximity, group sparsity, smoothness, DPP diversity
│   │   ├── metrics.py       # evaluate_cf_set — CF-quality evaluation
│   │   └── plot_basis.py    # basis-function figure
│   ├── baselines/
│   │   ├── interface.py     # common CounterfactualExplainer API
│   │   ├── comte.py         # CoMTE
│   │   └── forecast.py      # ForecastCF
│   ├── data_loader/         # AEP, CMAPSS (v2), IEEE PHM 2012 loaders
│   ├── models/              # per-dataset LSTM / GRU / CNN-LSTM / Transformer / STAR
│   ├── trainer/             # generic training loop + early stopping
│   ├── evaluation/          # canonical metric primitives and benchmarking
│   ├── utils/               # early stopping, plotting helpers
│   ├── train_cmapss_v3.py   # CMAPSS training entry point
│   ├── train_aep.py         # AEP training entry point
│   ├── train_transformer_ieee_phm.py   # IEEE PHM Transformer (Optuna HPO)
│   ├── train_gru_ieee_phm.py           # IEEE PHM GRU
│   └── train_bearing_phm.py            # IEEE PHM, raw-loader pipeline
├── notebooks/               # paper experiments, one per dataset
├── tests/                   # unit tests for the metric primitives
└── paper/                   # paper PDF, LaTeX source and figures
```

## Installation

```bash
git clone https://github.com/ghostshellV/basiscf.git
cd basiscf

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Requires Python 3.10+. A CUDA GPU is recommended for training but not required for
counterfactual generation on single queries.

All modules are imported as `src.*`, so run everything from the repository root
(`python -m src.train_cmapss_v3`, and start Jupyter from the root as well).

## Data

Datasets are **not** included in this repository. Download them and place them under `data/`:

| Dataset | Task | #Features | Seq. length | Source |
|---|---|---|---|---|
| NASA CMAPSS FD001–FD004 | RUL regression | 13 / 6 / 11 / 4 | 50 | NASA Prognostics Data Repository |
| IEEE PHM 2012 PRONOSTIA | RUL regression | 8 | 256 | IEEE PHM 2012 Data Challenge |
| AEP (Appliances Energy Prediction) | Forecasting | 21 | 12 | UCI ML Repository |

Expected default locations (all overridable via `--data-dir`):

```
data/processed/CMAPSS/data/        # train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt
data/processed/AEP/dataset/        # energydata_complete.csv
data/processed/ieee_phm/…/         # ieee_phm_sequences.npz, hyperparams.json, scaler.pkl
```

The IEEE PHM `.npz`/`hyperparams.json`/`scaler.pkl` artefacts are produced by
`notebooks/ieee_phm/01_IEEE_PHM_Data_Preprocess.ipynb` (windowing + feature extraction +
MinMax scaling of the raw PRONOSTIA vibration recordings).

## Training the black-box models

The counterfactual method treats the predictive model as fixed. The paper uses an encoder-only
Transformer for every dataset.

```bash
# CMAPSS — all four subsets
python -m src.train_cmapss_v3 --subsets FD001 FD002 FD003 FD004 \
                              --models transformer --seq-len 50

# AEP
python -m src.train_aep --models transformer

# IEEE PHM 2012 (Optuna hyper-parameter search, then final fit)
python -m src.train_transformer_ieee_phm
```

Checkpoints and metrics are written under `outputs/` (git-ignored).

## Generating counterfactuals

```python
import torch
from src.counterfactuals.core import (
    BasisGenerator, TSFeatureSchema, TargetSpec, GeneratorConfig, LossWeights,
)

schema = TSFeatureSchema(
    feature_names=feature_names,
    roles=["action"] * D,          # "immutable" | "action" | "state" | "context"
    min_vals=feature_mins,         # features are normalised to [0, 1] and clamped
    max_vals=feature_maxs,
    mad_inv=mad_inv,               # MAD-inverse proximity scaling
)

# Regression target expressed as a band, not a single point
target = TargetSpec(task_type="regression", target_value=100.0, target_range=(95.0, 105.0))

gen = BasisGenerator(
    model=model,                   # frozen, eval mode
    sequence_length=T,
    feature_dim=D,
    basis_type="bspline",          # bspline | fourier | rbf | polynomial | wavelet
    num_basis=8,                   # K
    device="cuda",
    config=GeneratorConfig(
        lr=0.02,
        max_iter=500,
        num_restarts=8,
        editable_roles=("action",),
    ),
)

weights = LossWeights(
    validity=1.0, proximity=0.50, sparsity=0.02,
    smoothness=0.15, diversity=0.25, channel_sparsity=0.20,
)

cfs, info = gen.generate(
    query_instance=x_query, target=target, schema=schema,
    num_cfs=3, loss_weights=weights,
)
```

Optimisation runs Adam followed by an LBFGS refinement, and selects the best counterfactual
lexicographically: validity first, then proximity, then channel sparsity, then smoothness.

`src/counterfactuals/usage.md` contains longer worked examples for both regression and
classification targets.

## Reproducing the paper experiments

Each notebook is self-contained end-to-end: load the trained model, build the feature schema,
sweep the basis families and basis sizes $K \in \{4, 6, 8, 10\}$, run both baselines, and emit the
evaluation tables and figures used in the paper.

| Notebook | Paper content |
|---|---|
| `notebooks/cmapss/03_CMAPSS_FD001_latest.ipynb` | CMAPSS FD001 results row |
| `notebooks/cmapss/03_CMAPSS_FD002_latest.ipynb` | CMAPSS FD002 results row |
| `notebooks/cmapss/03_CMAPSS_FD003_latest.ipynb` | FD003 row, full-engine-cycle plot, B-spline trajectories, validity–proximity trade-off |
| `notebooks/cmapss/03_CMAPSS_FD004_latest.ipynb` | CMAPSS FD004 results row |
| `notebooks/ieee_phm/01_IEEE_PHM_Data_Preprocess.ipynb` | PRONOSTIA preprocessing → `ieee_phm_sequences.npz` |
| `notebooks/ieee_phm/05_IEEE_PHM_Counterfactuals_final.ipynb` | IEEE PRONOSTIA row, perturbation heatmap, ablations |
| `notebooks/AEP/02_AEP_CF_new.ipynb` | AEP row and perturbation heatmap |

Counterfactual hyper-parameters per dataset (queries, $K$, editable roles, learning rate,
iterations, restarts, loss weights) are listed in Table 4 of the paper and set at the top of each
notebook.

## Baselines

| Baseline | Description | File |
|---|---|---|
| **CoMTE** | Counterfactual multivariate time-series explanation by segment replacement from the training set | `src/baselines/comte.py` |
| **ForecastCF** | Gradient-based counterfactual generation for time-series forecasting, optimising raw time steps toward a target band | `src/baselines/forecast.py` |

Both implement the same `CounterfactualExplainer` interface as `BasisGenerator`, so they are
evaluated through an identical code path.

## Evaluation metrics

Counterfactual quality (`src/counterfactuals/metrics.py`, primitives in `src/evaluation/metrics.py`):

- **Validity** — residual violation of the target band, plus binary success rate
- **Proximity** — MAD-scaled per-element L1 and L2 distance to the query
- **Group sparsity** — fraction of channels edited
- **Smoothness** — second-difference energy of the perturbation
- **Diversity** — mean pairwise distance across the returned counterfactual set

Model quality (`src/evaluation/`): RMSE, MAE, R², NASA prognostics score.

```bash
python -m pytest tests/          # unit tests for the metric primitives
```

## Citation

```bibtex
@inproceedings{basiscf,
  title     = {Basis-Guided Counterfactual Generation for Explainable Multivariate Time Series Models},
  author    = {},
  year      = {2026}
}
```

Developed at .
