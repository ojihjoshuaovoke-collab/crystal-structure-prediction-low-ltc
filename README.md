# LTC & Heat Capacity Predictor

A deployed, interactive tool built on real DFT data from my own published research:

> J. Ojih, M. Al-Fahdi, Y. Yao, J. Hu, M. Hu. "Graph theory and graph
> neural network assisted high-throughput crystal structure prediction
> and screening for energy conversion and storage." *J. Mater. Chem.
> A*, 2024, 12, 8502-8515. https://doi.org/10.1039/D3TA06190F

> J. Ojih, C. Shen, A. Rodriguez, H. Zhang, K. Choudhary, M. Hu.
> "High-throughput computational discovery of 3218 ultralow thermal
> conductivity and dynamically stable materials by dual machine
> learning models." *J. Mater. Chem. A*, 2023, 11, 24169-24183.
> https://doi.org/10.1039/D3TA04874H

> J. Ojih, U. Onyekpe, A. Rodriguez, J. Hu, C. Peng, M. Hu. "Machine
> Learning Accelerated Discovery of Promising Thermal Energy Storage
> Materials with High Heat Capacity." *ACS Appl. Mater. Interfaces*,
> 2022, 14, 43277-43289. https://doi.org/10.1021/acsami.2c11350

**Repo:** https://github.com/ojihjoshuaovoke-collab/crystal-structure-prediction-low-ltc
**Live demo:** _(add your Streamlit Cloud link here after redeploying)_

## What this is

Two Random Forest models, trained on **4706 real DFT-calculated
structures** (`data/combined_features.csv`), predicting:

- **Lattice thermal conductivity (LTC)** — log10-scale, matching the
  training approach used in the original papers
- **Heat capacity** — raw units (J/mol·K)

Unlike an earlier version of this project (which trained on a GNN's
*predictions* on 646 newly-discovered structures — a copy of a
model's output, not ground truth), this version trains directly on
real DFT-calculated values — the same targets the original papers'
graph neural networks (ALIGNN, deeperGATGNN) were trained on.

## Real results (held-out 20% test split)

```
LTC:            R² = 0.761 (log10 scale), MAE = 0.269
                 -> predictions within ~1.86x of true value on average
Heat capacity:   R² = 0.997, MAE = 1.67 J/mol-K
```

For comparison, the original papers' graph neural networks (which
learn directly from the full crystal graph, not summary descriptors)
achieve R² = 0.834 for LTC (Ojih et al. 2023) and R² = 0.998 for heat
capacity (Ojih et al. 2022). A Random Forest on 11 hand-picked
descriptors reaching R² = 0.76 for LTC and R² = 0.997 for heat
capacity is a strong, honest result for a much simpler model —
particularly for heat capacity, where `Number of Atom` alone accounts
for 95% of feature importance, consistent with the Dulong-Petit
relationship (C ≈ 3NR) both papers discuss.

## The Dulong-Petit limit

Heat capacity predictions are shown alongside the classical
Dulong-Petit limit (3NR) and their ratio to it. A ratio above 1 is
rare but **not automatically an error** — Ojih et al. 2022 found a
real, DFT-verified example (MnIn₂Se₄) that genuinely exceeds the
limit near room temperature, driven by a phonon band-gap/optical-
branch effect confirmed via full phonon dispersion calculations. The
app flags ratio > 1 as worth closer inspection, not as invalid.

## App

`app.py` (Streamlit) has two modes:

1. **Upload a CIF** — accurate, structure-aware predictions using real
   descriptors (volume, density, bond length) extracted via `pymatgen`.
2. **Formula only** — a rough estimate for compositions with no known
   structure; structural descriptors are backfilled with dataset
   medians and the estimate is clearly flagged as lower-confidence.

## Setup

```bash
pip install -r requirements.txt
python train_real_model.py   # trains both models on the 4706-structure dataset
streamlit run app.py         # launch locally
```

The app also auto-trains on first run if the model files are missing
(e.g. a fresh deployment), so this works out of the box on Streamlit
Cloud without a separate training step.

## Deploying

Push to GitHub, then deploy via [Streamlit Community Cloud](https://share.streamlit.io)
(free, no GPU/compute paywall) pointing at `app.py`.

## Repo structure

```
materials-ltc-app/
├── app.py                    # Streamlit app (CIF upload + formula modes)
├── train_real_model.py       # trains both models on real DFT data
├── real_features.py          # feature extraction (structure or formula)
├── data/
│   └── combined_features.csv # 4706 real structures: descriptors + DFT LTC/heat capacity
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Attribution

Built on my own peer-reviewed, first-author publications, using real
DFT-calculated data from that research. This is an independent,
simplified Random Forest surrogate of the original papers' graph
neural network models, built for portfolio/deployment purposes — not
a claim of equivalent accuracy to the original GNNs.
