"""
Streamlit app: LTC & Heat Capacity Predictor

Trained on real DFT ground truth from my own published research:
- Ojih et al., J. Mater. Chem. A, 2024, 12, 8502 (crystal structure discovery)
- Ojih et al., J. Mater. Chem. A, 2023, 11, 24169 (LTC screening, P3/MSD descriptors)
- Ojih et al., ACS Appl. Mater. Interfaces, 2022, 14, 43277 (heat capacity, Dulong-Petit)

Two Random Forest models trained on 4706 real DFT-calculated structures:
one predicts lattice thermal conductivity (LTC), one predicts heat
capacity. Upload a CIF for accurate, structure-aware predictions, or
enter a formula for a rough composition-only estimate.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tempfile

from pymatgen.core import Structure
from real_features import features_from_structure, features_from_formula
from train_real_model import train_all

st.set_page_config(page_title="LTC & Heat Capacity Predictor", page_icon="🔬")

st.title("🔬 LTC & Heat Capacity Predictor")
st.caption(
    "Trained on 4706 real DFT-calculated structures from my published research on "
    "[crystal structure discovery](https://doi.org/10.1039/D3TA06190F), "
    "[LTC screening](https://doi.org/10.1039/D3TA04874H), and "
    "[thermal energy storage materials](https://doi.org/10.1021/acsami.2c11350)."
)

LTC_MODEL_PATH = "model_ltc_real.joblib"
HC_MODEL_PATH = "model_hc_real.joblib"
DATA_PATH = "data/combined_features.csv"

if not (os.path.exists(LTC_MODEL_PATH) and os.path.exists(HC_MODEL_PATH)):
    with st.spinner("First-time setup: training models on 4706 real structures (~30-60s)..."):
        train_all(data_path=DATA_PATH, ltc_out=LTC_MODEL_PATH, hc_out=HC_MODEL_PATH)

ltc_bundle = joblib.load(LTC_MODEL_PATH)
hc_bundle = joblib.load(HC_MODEL_PATH)
df = pd.read_csv(DATA_PATH)
medians = df[["volume", "Number Density", "Mass Density", "Bond Length"]].median().to_dict()

st.success(
    f"LTC model: R\u00b2 = {ltc_bundle['test_r2']:.2f} (log scale), "
    f"Heat capacity model: R\u00b2 = {hc_bundle['test_r2']:.3f} \u2014 "
    f"both evaluated on a held-out 20% split of the real 4706-structure dataset.",
    icon="\u2705",
)

R_GAS = 8.314  # J/mol-K


def predict_and_display(features: dict, accurate: bool):
    X_ltc = np.array([[features[f] for f in ltc_bundle["features"]]])
    X_hc = np.array([[features[f] for f in hc_bundle["features"]]])

    log_ltc_pred = ltc_bundle["model"].predict(X_ltc)[0]
    ltc_pred = 10 ** log_ltc_pred
    hc_pred = hc_bundle["model"].predict(X_hc)[0]

    n_atoms = features["Number of Atom"]
    dulong_petit = 3 * n_atoms * R_GAS
    ratio = hc_pred / dulong_petit

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted LTC", f"{ltc_pred:.2f} W/m\u00b7K")
    col2.metric("Predicted heat capacity", f"{hc_pred:.1f} J/mol\u00b7K")
    col3.metric("Ratio to Dulong-Petit limit", f"{ratio:.2f}")

    if ltc_pred < 1.0:
        st.info("Ultralow LTC \u2014 potential thermoelectric candidate.", icon="\U0001F9CA")

    if ratio > 1.0:
        st.warning(
            "Predicted heat capacity exceeds the classical Dulong-Petit limit. "
            "This is rare but not necessarily wrong \u2014 a real example (MnIn\u2082Se\u2084) was "
            "confirmed by full phonon dispersion calculations in Ojih et al. 2022, driven by "
            "a phonon band-gap/optical-branch effect. Treat this as a flag for closer "
            "inspection (e.g. phonon dispersion), not an automatic error.",
            icon="\u26A0\ufe0f",
        )

    if not accurate:
        st.caption(
            "Composition-only estimate \u2014 structural descriptors (volume, density, bond "
            "length) were backfilled with dataset medians since no structure was provided. "
            "Upload a CIF for an accurate, structure-aware prediction."
        )


tab1, tab2 = st.tabs(["\U0001F4C4 Upload a CIF (accurate)", "\U0001F9EA Formula only (rough estimate)"])

with tab1:
    st.write("Upload a CIF file for accurate, structure-aware predictions.")
    uploaded = st.file_uploader("CIF file", type=["cif"])
    if uploaded is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            struct = Structure.from_file(tmp_path)
            st.write(f"Parsed structure: **{struct.composition.reduced_formula}**, {struct.num_sites} atoms")
            features = features_from_structure(struct)
            predict_and_display(features, accurate=True)
        except Exception as e:
            st.error(f"Could not parse CIF: {e}")

with tab2:
    formula = st.text_input("Chemical formula:", value="YTl3")
    if formula:
        try:
            features = features_from_formula(formula, medians)
            predict_and_display(features, accurate=False)
        except Exception as e:
            st.error(f"Could not parse formula: {e}")

st.divider()
st.caption(
    "Models are Random Forests trained on 11 hand-picked descriptors (composition + "
    "structure), not a reproduction of the original papers' graph neural networks "
    "(ALIGNN/deeperGATGNN), which achieve higher accuracy by learning directly from the "
    "full crystal graph. Built as an independent, deployable surrogate for portfolio/"
    "demonstration purposes."
)
