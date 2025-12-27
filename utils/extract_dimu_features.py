#!/usr/bin/env python3
"""
extract_dimu_features.py
  rec_dimu_mu_pos_px, rec_dimu_mu_pos_py, rec_dimu_mu_pos_pz
  rec_dimu_mu_neg_px, rec_dimu_mu_neg_py, rec_dimu_mu_neg_pz
  rec_track_pos_x_st1, rec_track_neg_x_st1, rec_track_pos_px_st1, rec_track_neg_px_st1
  rec_track_pos_vz, rec_track_neg_vz
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import ROOT
from ROOT import TLorentzVector

MUON_MASS_GEV = 0.1056

REQUIRED_BRANCHES = [
    # Muon momenta
    "rec_dimu_mu_pos_px", "rec_dimu_mu_pos_py", "rec_dimu_mu_pos_pz",
    "rec_dimu_mu_neg_px", "rec_dimu_mu_neg_py", "rec_dimu_mu_neg_pz",
    # Station-1
    "rec_track_pos_x_st1", "rec_track_neg_x_st1",
    "rec_track_pos_px_st1", "rec_track_neg_px_st1",
    # Vertex z
    "rec_track_pos_vz", "rec_track_neg_vz",
]

FEATURE_NAMES: List[str] = [
    "rec_dimu_y",
    "rec_dimu_eta",
    "rec_dimu_E",
    "rec_dimu_pz",
    "rec_dimu_M",
    "rec_mu_theta_pos",
    "rec_mu_theta_neg",
    "rec_mu_open_angle",
    "rec_mu_dpt",
    "rec_dimu_mT",
    "rec_mu_Epos",
    "rec_mu_Eneg",
    "rec_track_pos_x_st1",
    "rec_track_neg_x_st1",
    "rec_track_pos_px_st1",
    "rec_track_neg_px_st1",
    "rec_dz_vtx",
    "rec_mu_deltaR",
]


def delta_phi(phi1: float, phi2: float) -> float:
    """Return (phi1 - phi2) wrapped into [-pi, pi]."""
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _open_tree(input_file: Path | str, tree_name: str):
    f = ROOT.TFile(str(input_file), "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {input_file}")
    t = f.Get(tree_name)
    if not t:
        f.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in ROOT file: {input_file}")
    return f, t


def _has_branch(tree, name: str) -> bool:
    br_list = tree.GetListOfBranches()
    return bool(br_list.FindObject(name))


def _require_branches(tree, names: List[str]):
    missing = [b for b in names if not _has_branch(tree, b)]
    if missing:
        raise RuntimeError(f"Missing required branches in '{tree.GetName()}': {missing}")


def extract_features(
    input_file: Path | str,
    *,
    tree_name: str = "tree",
    output_path: Optional[Path | str] = None,
    mass_min: float = 1.0,
    mass_max: float = 6.0,
    verbose_every: int = 10000,
    dtype=np.float64,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Returns:
      X            : (N, 18) array
      feature_names: list of feature names (length 18)
      meta         : dict with provenance
    """
    f, tree = _open_tree(input_file, tree_name)

    try:
        _require_branches(tree, REQUIRED_BRANCHES)

        n_entries = int(tree.GetEntries())
        feats: List[List[float]] = []

        mu_pos = TLorentzVector()
        mu_neg = TLorentzVector()

        kept_mass = 0
        skipped_zero = 0

        for i, event in enumerate(tree):
            if verbose_every and (i % verbose_every == 0):
                print(f"Processing event {i}/{n_entries}")

            # Read muon momenta
            pxp = float(getattr(event, "rec_dimu_mu_pos_px"))
            pyp = float(getattr(event, "rec_dimu_mu_pos_py"))
            pzp = float(getattr(event, "rec_dimu_mu_pos_pz"))

            pxn = float(getattr(event, "rec_dimu_mu_neg_px"))
            pyn = float(getattr(event, "rec_dimu_mu_neg_py"))
            pzn = float(getattr(event, "rec_dimu_mu_neg_pz"))

            mu_pos.SetXYZM(pxp, pyp, pzp, MUON_MASS_GEV)
            mu_neg.SetXYZM(pxn, pyn, pzn, MUON_MASS_GEV)

            dimu = mu_pos + mu_neg
            m = float(dimu.M())
            if (m < mass_min) or (m > mass_max):
                continue
            kept_mass += 1

            # Dimuon
            dimu_y = float(dimu.Rapidity())
            dimu_eta = float(dimu.Eta())
            dimu_E = float(dimu.E())
            dimu_pz = float(dimu.Pz())
            dimu_pt = float(dimu.Pt())
            dimu_mT = float(np.sqrt(m * m + dimu_pt * dimu_pt))

            # Opening angle
            vpos = mu_pos.Vect()
            vneg = mu_neg.Vect()
            denom = float(vpos.Mag() * vneg.Mag())
            if denom <= 0:
                skipped_zero += 1
                continue
            cos_open = float(vpos.Dot(vneg) / denom)
            open_angle = float(np.arccos(np.clip(cos_open, -1.0, 1.0)))

            # Single-muon derived
            theta_pos = float(np.arctan2(mu_pos.Pt(), mu_pos.Pz()))
            theta_neg = float(np.arctan2(mu_neg.Pt(), mu_neg.Pz()))
            dpt = float(mu_pos.Pt() - mu_neg.Pt())
            Epos = float(mu_pos.E())
            Eneg = float(mu_neg.E())

            # Station-1 (required)
            st1_x_pos = float(getattr(event, "rec_track_pos_x_st1"))
            st1_x_neg = float(getattr(event, "rec_track_neg_x_st1"))
            st1_px_pos = float(getattr(event, "rec_track_pos_px_st1"))
            st1_px_neg = float(getattr(event, "rec_track_neg_px_st1"))

            # dz_vtx (required)
            zpos = float(getattr(event, "rec_track_pos_vz"))
            zneg = float(getattr(event, "rec_track_neg_vz"))
            dz_vtx = float(zpos - zneg)

            # ΔR
            eta_pos = float(mu_pos.Eta())
            eta_neg = float(mu_neg.Eta())
            d_eta = float(eta_pos - eta_neg)
            d_phi = float(delta_phi(mu_pos.Phi(), mu_neg.Phi()))
            deltaR = float(np.sqrt(d_eta * d_eta + d_phi * d_phi))

            feats.append([
                dimu_y, dimu_eta, dimu_E, dimu_pz, m,
                theta_pos, theta_neg, open_angle,
                dpt, dimu_mT, Epos, Eneg,
                st1_x_pos, st1_x_neg, st1_px_pos, st1_px_neg,
                dz_vtx,
                deltaR,
            ])

        X = np.asarray(feats, dtype=dtype)

        meta: Dict[str, Any] = {
            "input_file": str(input_file),
            "tree_name": tree_name,
            "mass_min": mass_min,
            "mass_max": mass_max,
            "muon_mass_GeV": MUON_MASS_GEV,
            "n_entries_total": n_entries,
            "n_entries_pass_mass": kept_mass,
            "n_entries_skipped_zero_vectors": skipped_zero,
            "n_rows_output": int(X.shape[0]),
            "n_features": int(X.shape[1]) if X.ndim == 2 else None,
            "feature_names": FEATURE_NAMES,
            "required_branches": REQUIRED_BRANCHES,
        }

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                X=X,
                feature_names=np.asarray(FEATURE_NAMES, dtype=object),
                meta_json=json.dumps(meta),
            )
            print(f"Saved: {out}")
            print(f"Shape: {X.shape}")

        return X, FEATURE_NAMES, meta

    finally:
        f.Close()

def main():
    p = argparse.ArgumentParser(description="Extract dimuon features from a ROOT file (strict standardized branches).")
    p.add_argument("--input", required=True, type=Path, help="Input ROOT file")
    p.add_argument("--tree", default="tree", help="TTree name (default: tree)")
    p.add_argument("--out", required=True, type=Path, help="Output .npz path")
    p.add_argument("--mass-min", type=float, default=2.0, help="Mass window min (GeV)")
    p.add_argument("--mass-max", type=float, default=4.0, help="Mass window max (GeV)")
    p.add_argument("--verbose-every", type=int, default=10000, help="Print progress every N events (0 disables)")
    args = p.parse_args()

    extract_features(
        args.input,
        tree_name=args.tree,
        output_path=args.out,
        mass_min=args.mass_min,
        mass_max=args.mass_max,
        verbose_every=args.verbose_every,
    )

if __name__ == "__main__":
    main()

