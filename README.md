# LTC & Heat Capacity Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ojihjoshuaovoke-collab-crystal-structure-prediction--app-yc8syi.streamlit.app)

A deployed, interactive tool built on real DFT data from my own published research:

> J. Ojih, M. Al-Fahdi, Y. Yao, J. Hu, M. Hu. "Graph theory and graph
> neural network assisted high-throughput crystal structure prediction
> and screening for energy conversion and storage." *J. Mater. Chem.
> A*, 2024, 12, 8502-8515. https://doi.org/10.1039/D3TA06190F

**Repo:** https://github.com/ojihjoshuaovoke-collab/crystal-structure-prediction-low-ltc
**Live demo:** [Try it live \u2192 predict LTC & heat capacity from a CIF or formula](https://ojihjoshuaovoke-collab-crystal-structure-prediction--app-yc8syi.streamlit.app)

## What this is

Two Random Forest models, trained on **4706 real DFT-calculated
structures** (`data/combined_features.csv`) from the paper above,
predicting:

- **Lattice thermal conductivity (LTC)** — log10-scale, matching the
  training approach used in the original paper
- **Heat capacity** — raw units (J/mol·K)

Unlike an earlier version of this project (which trained on a GNN's
*predictions* on 646 newly-discovered structures — a copy of a
model's output, not ground truth), this version trains directly on
real DFT-calculated values — the same targets the original paper's
graph neural networks (ALIGNN, OGCNN, deeperGATGNN) were trained on.

## Real results (held-out 20% test split)

```
LTC:            R² = 0.761 (log10 scale), MAE = 0.269
                 -> predictions within ~1.86x of true value on average
Heat capacity:   R² = 0.997, MAE = 1.67 J/mol-K
```

A Random Forest on 11 hand-picked descriptors reaching R² = 0.76 for
LTC and R² = 0.997 for heat capacity is a strong, honest result for a
much simpler model than the original paper's graph neural networks
(which learn directly from the full crystal graph, not summary
descriptors). Heat capacity is notably easier to predict —
`Number of Atom` alone accounts for 95% of feature importance,
consistent with the Dulong-Petit relationship (C ≈ 3NR).

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

Built on my own peer-reviewed, first-author publication (CC-BY-NC
4.0), using real DFT-calculated data from that research. This is an
independent, simplified Random Forest surrogate of the original
paper's graph neural network models, built for portfolio/deployment
purposes — not a claim of equivalent accuracy to the original GNNs.
