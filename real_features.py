"""
Extract the 11 features used by the real-data models (train_real_model.py)
from a new crystal structure (CIF) or, as a rough fallback, from a
chemical formula alone.

Feature names match data/combined_features.csv exactly:
    Total Weight, volume, Average Weight, Number Density, Mass Density,
    Bond Length, Number of Atom, Number of unpaired Electron,
    Average Number of Electron, Maximun Principle quantum number,
    Pauling Electronegativity
"""

from __future__ import annotations
import numpy as np
from pymatgen.core import Structure, Composition
from mendeleev import element as mendeleev_element

CUTOFF = 10.0
DELR = 0.1201


def features_from_structure(struct: Structure) -> dict:
    """Full, accurate feature set from a real parsed crystal structure."""
    n_atoms = struct.num_sites
    total_weight = struct.composition.weight
    volume = struct.volume
    mass_density = struct.density
    number_density = n_atoms / volume
    avg_weight = total_weight / n_atoms

    # Average first-neighbor bond length (same method as extract_all_features.py)
    bond_length_total = 0.0
    n_counted = 0
    for site in struct.sites:
        neighbors = struct.get_neighbors(site, r=CUTOFF)
        if not neighbors:
            continue
        dists = sorted(nb.nn_distance for nb in neighbors)
        d0 = dists[0]
        first_shell = [d for d in dists if abs(d - d0) < DELR]
        bond_length_total += float(np.mean(first_shell))
        n_counted += 1
    bond_length = bond_length_total / n_counted if n_counted else np.nan

    # Electronic descriptors, per-atom averaged (mendeleev)
    unpaired, n_elec, max_n, pauling = [], [], [], []
    for site in struct.sites:
        el = mendeleev_element(site.specie.symbol)
        unpaired.append(el.ec.unpaired_electrons())
        n_elec.append(el.ec.ne())
        max_n.append(el.ec.max_n())
        x = el.electronegativity(scale="pauling")
        if x is not None:
            pauling.append(x)

    return {
        "Total Weight": total_weight,
        "volume": volume,
        "Average Weight": avg_weight,
        "Number Density": number_density,
        "Mass Density": mass_density,
        "Bond Length": bond_length,
        "Number of Atom": n_atoms,
        "Number of unpaired Electron": float(np.mean(unpaired)),
        "Average Number of Electron": float(np.mean(n_elec)),
        "Maximun Principle quantum number": float(np.mean(max_n)),
        "Pauling Electronegativity": float(np.mean(pauling)) if pauling else np.nan,
    }


def features_from_formula(formula: str, medians: dict) -> dict:
    """
    Rough fallback for a formula with no known structure: composition-only
    features are computed properly; structural features (volume, density,
    bond length, number density) are backfilled with dataset medians since
    they can't be known without an actual structure.
    """
    comp = Composition(formula)
    elements = comp.elements
    n_atoms = comp.num_atoms
    total_weight = comp.weight
    avg_weight = total_weight / n_atoms

    unpaired, n_elec, max_n, pauling = [], [], [], []
    for el in elements:
        frac = comp.get_atomic_fraction(el)
        m_el = mendeleev_element(el.symbol)
        unpaired.append(m_el.ec.unpaired_electrons() * frac)
        n_elec.append(m_el.ec.ne() * frac)
        max_n.append(m_el.ec.max_n() * frac)
        x = m_el.electronegativity(scale="pauling")
        if x is not None:
            pauling.append(x * frac)

    return {
        "Total Weight": total_weight,
        "volume": medians["volume"],
        "Average Weight": avg_weight,
        "Number Density": medians["Number Density"],
        "Mass Density": medians["Mass Density"],
        "Bond Length": medians["Bond Length"],
        "Number of Atom": n_atoms,
        "Number of unpaired Electron": sum(unpaired),
        "Average Number of Electron": sum(n_elec),
        "Maximun Principle quantum number": sum(max_n),
        "Pauling Electronegativity": sum(pauling) if pauling else np.nan,
    }
