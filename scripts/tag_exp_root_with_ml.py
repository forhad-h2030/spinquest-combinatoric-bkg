#!/usr/bin/env python3
from pathlib import Path
import sys
import math
import array
import json
from typing import Optional, Tuple

import numpy as np
import torch
import ROOT
from ROOT import TLorentzVector

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../spinquest-combinatoric-bkg
sys.path.insert(0, str(REPO_ROOT))

from utils.core_train_binary import ParticleClassifierBinary  # must match your training model class
MUON_MASS_GEV = 0.1056
REQUIRED_BRANCHES = [
    "rec_dimu_mu_pos_px", "rec_dimu_mu_pos_py", "rec_dimu_mu_pos_pz",
    "rec_dimu_mu_neg_px", "rec_dimu_mu_neg_py", "rec_dimu_mu_neg_pz",
    "rec_track_pos_x_st1", "rec_track_neg_x_st1",
    "rec_track_pos_px_st1", "rec_track_neg_px_st1",
    "rec_track_pos_vz", "rec_track_neg_vz",
]

# ml_class encoding
CLASS_JPSI = 1
CLASS_PSIP = 2
CLASS_DY   = 3
CLASS_COMB = 4


def delta_phi(phi1: float, phi2: float) -> float:
    """Return (phi1 - phi2) wrapped into [-pi, pi]."""
    dphi = phi1 - phi2
    return (dphi + math.pi) % (2.0 * math.pi) - math.pi


def require_branches(tree, names):
    br_list = tree.GetListOfBranches()
    missing = [b for b in names if not br_list.FindObject(b)]
    if missing:
        raise RuntimeError(f"Missing required branches in tree '{tree.GetName()}': {missing}")


def compute_18_features(event) -> Optional[np.ndarray]:
    """
    Compute the same 18 features used in training (float32).
    Returns None if the event is invalid (e.g., zero magnitude vectors for opening angle).
    """
    pxp = float(getattr(event, "rec_dimu_mu_pos_px"))
    pyp = float(getattr(event, "rec_dimu_mu_pos_py"))
    pzp = float(getattr(event, "rec_dimu_mu_pos_pz"))

    pxn = float(getattr(event, "rec_dimu_mu_neg_px"))
    pyn = float(getattr(event, "rec_dimu_mu_neg_py"))
    pzn = float(getattr(event, "rec_dimu_mu_neg_pz"))

    mu_pos = TLorentzVector()
    mu_neg = TLorentzVector()
    mu_pos.SetXYZM(pxp, pyp, pzp, MUON_MASS_GEV)
    mu_neg.SetXYZM(pxn, pyn, pzn, MUON_MASS_GEV)

    dimu = mu_pos + mu_neg
    m = float(dimu.M())

    # Opening angle guard
    vpos = mu_pos.Vect()
    vneg = mu_neg.Vect()
    denom = float(vpos.Mag() * vneg.Mag())
    if denom <= 0:
        return None

    cos_open = float(vpos.Dot(vneg) / denom)
    open_angle = float(math.acos(max(-1.0, min(1.0, cos_open))))

    # Dimuon
    dimu_y = float(dimu.Rapidity())
    dimu_eta = float(dimu.Eta())
    dimu_E = float(dimu.E())
    dimu_pz = float(dimu.Pz())
    dimu_pt = float(dimu.Pt())
    dimu_mT = float(math.sqrt(m * m + dimu_pt * dimu_pt))

    # Single muon derived
    theta_pos = float(math.atan2(mu_pos.Pt(), mu_pos.Pz()))
    theta_neg = float(math.atan2(mu_neg.Pt(), mu_neg.Pz()))
    dpt = float(mu_pos.Pt() - mu_neg.Pt())
    Epos = float(mu_pos.E())
    Eneg = float(mu_neg.E())

    # Station-1
    st1_x_pos = float(getattr(event, "rec_track_pos_x_st1"))
    st1_x_neg = float(getattr(event, "rec_track_neg_x_st1"))
    st1_px_pos = float(getattr(event, "rec_track_pos_px_st1"))
    st1_px_neg = float(getattr(event, "rec_track_neg_px_st1"))

    # dz_vtx
    zpos = float(getattr(event, "rec_track_pos_vz"))
    zneg = float(getattr(event, "rec_track_neg_vz"))
    dz_vtx = float(zpos - zneg)

    # deltaR
    eta_pos = float(mu_pos.Eta())
    eta_neg = float(mu_neg.Eta())
    d_eta = float(eta_pos - eta_neg)
    d_phi = float(delta_phi(mu_pos.Phi(), mu_neg.Phi()))
    deltaR = float(math.sqrt(d_eta * d_eta + d_phi * d_phi))

    return np.array([
        dimu_y, dimu_eta, dimu_E, dimu_pz, m,
        theta_pos, theta_neg, open_angle,
        dpt, dimu_mT, Epos, Eneg,
        st1_x_pos, st1_x_neg, st1_px_pos, st1_px_neg,
        dz_vtx,
        deltaR,
    ], dtype=np.float32)


def apply_scaler(x18: np.ndarray, scaler) -> np.ndarray:
    """
    scaler: None or dict with keys {"mean","std"} matching feature dim.
    """
    if scaler is None:
        return x18
    mu = np.asarray(scaler["mean"], dtype=np.float32).reshape(-1)
    sd = np.asarray(scaler["std"], dtype=np.float32).reshape(-1)
    return ((x18 - mu) / sd).astype(np.float32)


def load_payload_model(ckpt_path: Path, device: str) -> Tuple[torch.nn.Module, Optional[dict]]:
    """
    Loads a 'payload' checkpoint dict saved by core_train_binary.py:
      { "state_dict": ..., "input_dim": int, "cfg": {...}, "scaler": None|{mean,std}, ... }

    Returns:
      model (eval mode), scaler (or None)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "input_dim" in ckpt):
        raise RuntimeError(
            f"{ckpt_path} is not a payload checkpoint. "
            f"Expected dict with keys: state_dict, input_dim, cfg, scaler."
        )

    input_dim = int(ckpt["input_dim"])
    cfg = ckpt.get("cfg", {})

    hidden_dim = int(cfg.get("hidden_dim", 512))
    num_layers = int(cfg.get("num_layers", 4))
    dropout_rate = float(cfg.get("dropout_rate", 0.3))

    model = ParticleClassifierBinary(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    scaler = ckpt.get("scaler", None)
    return model, scaler


@torch.no_grad()
def prob_pos(model: torch.nn.Module, x18: np.ndarray, device: str) -> float:
    """
    Predict p(POS) for a single (18,) vector.
    """
    x = torch.tensor(x18.reshape(1, -1), dtype=torch.float32, device=device)
    logit = model(x).squeeze(0)
    return float(torch.sigmoid(logit).item())


def main():
    import argparse

    # ---- defaults (override as needed) ----
    default_in  = REPO_ROOT / "data" / "raw_input" / "exp_tgt_data.root"
    default_out = REPO_ROOT / "data" / "raw_input" / "exp_tagged_tgt_data.root"

    default_ckpt_jpsi = REPO_ROOT / "models" / "jpsi_vs_nonjpsi.best.pth"
    default_ckpt_psip = REPO_ROOT / "models" / "psip_vs_nonpsip.best.pth"
    default_ckpt_dy   = REPO_ROOT / "models" / "dy_comb_raw.best.pth"  

    example_cmd = f"""Example:
  python3 {Path(__file__).name} \\
    --input  {default_in} \\
    --tree   tree \\
    --output {default_out} \\
    --mass-min 2.0 --mass-max 5.0 \\
    --thr 0.80 \\
    --ckpt-jpsi {default_ckpt_jpsi} \\
    --ckpt-psip {default_ckpt_psip} \\
    --ckpt-dy   {default_ckpt_dy}
"""

    p = argparse.ArgumentParser(
        description="Tag an experimental ROOT tree with ML branches: ml_class + ml_p_*.",
        epilog=example_cmd,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("--input", type=Path, default=default_in,
                   help=f"Input ROOT file (default: {default_in})")
    p.add_argument("--tree", default="tree",
                   help="TTree name (default: tree)")
    p.add_argument("--output", type=Path, default=default_out,
                   help=f"Output ROOT file (default: {default_out})")

    p.add_argument("--mass-min", type=float, default=2.0,
                   help="Dimuon mass window min (GeV) (default: 2.0)")
    p.add_argument("--mass-max", type=float, default=5.0,
                   help="Dimuon mass window max (GeV) (default: 5.0)")
    p.add_argument("--thr", type=float, default=0.80,
                   help="Threshold for POS decisions in each stage (default: 0.80)")

    p.add_argument("--ckpt-jpsi", type=Path, default=default_ckpt_jpsi,
                   help=f"Checkpoint for J/psi vs non-psi (outputs p(J/psi)) (default: {default_ckpt_jpsi})")
    p.add_argument("--ckpt-psip", type=Path, default=default_ckpt_psip,
                   help=f"Checkpoint for psip vs non-psip (outputs p(psip)) (default: {default_ckpt_psip})")
    p.add_argument("--ckpt-dy", type=Path, default=default_ckpt_dy,
                   help=f"Checkpoint for DY vs comb (outputs p(DY)) (default: {default_ckpt_dy})")

    args = p.parse_args()

    # ---- sanity checks ----
    missing = []
    if not args.input.exists(): missing.append(f"--input {args.input}")
    if not args.ckpt_jpsi.exists(): missing.append(f"--ckpt-jpsi {args.ckpt_jpsi}")
    if not args.ckpt_psip.exists(): missing.append(f"--ckpt-psip {args.ckpt_psip}")
    if not args.ckpt_dy.exists(): missing.append(f"--ckpt-dy {args.ckpt_dy}")

    if missing:
        print("[ERROR] Some required files were not found:")
        for m in missing:
            print("  ", m)
        print("\nRun with --help to see an example command.\n")
        raise SystemExit(2)

    thr = float(args.thr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device} thr={thr}")

    # ---- load models ----
    model_jpsi, scaler_jpsi = load_payload_model(args.ckpt_jpsi, device)
    model_psip, scaler_psip = load_payload_model(args.ckpt_psip, device)
    model_dy,   scaler_dy   = load_payload_model(args.ckpt_dy, device)

    fin = ROOT.TFile.Open(str(args.input), "READ")
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Could not open {args.input}")
    tin = fin.Get(args.tree)
    if not tin:
        raise RuntimeError(f"Tree '{args.tree}' not found in {args.input}")

    require_branches(tin, REQUIRED_BRANCHES)

    # ---- create output tagged tree ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fout = ROOT.TFile.Open(str(args.output), "RECREATE")
    tout = tin.CloneTree(0)  # structure only

    # new branches
    b_class = array.array("i", [0])
    b_pjpsi = array.array("f", [0.0])
    b_ppsip = array.array("f", [0.0])
    b_pdy   = array.array("f", [0.0])
    b_pcomb = array.array("f", [0.0])

    tout.Branch("ml_class", b_class, "ml_class/I")
    tout.Branch("ml_p_jpsi", b_pjpsi, "ml_p_jpsi/F")
    tout.Branch("ml_p_psip", b_ppsip, "ml_p_psip/F")
    tout.Branch("ml_p_dy",   b_pdy,   "ml_p_dy/F")
    tout.Branch("ml_p_comb", b_pcomb, "ml_p_comb/F")

    n_in = int(tin.GetEntries())
    kept_mass = 0
    skipped_zero = 0
    n_written = 0

    for i in range(n_in):
        tin.GetEntry(i)

        feats = compute_18_features(tin)
        if feats is None:
            skipped_zero += 1
            continue

        m = float(feats[4])  # rec_dimu_M
        if (m < args.mass_min) or (m > args.mass_max):
            continue
        kept_mass += 1

        #: reset outputs every event so no stale values leak through
        b_class[0] = 0
        b_pjpsi[0] = 0.0
        b_ppsip[0] = 0.0
        b_pdy[0]   = 0.0
        b_pcomb[0] = 0.0

        # (1) J/psi vs non-psi: outputs p(J/psi)
        pj = prob_pos(model_jpsi, apply_scaler(feats, scaler_jpsi), device)
        b_pjpsi[0] = pj

        if pj >= thr:
            b_class[0] = CLASS_JPSI
            # keep others at 0.0
            tout.Fill()
            n_written += 1
            continue

        # (2) psip vs non-psip: outputs p(psip)
        pp = prob_pos(model_psip, apply_scaler(feats, scaler_psip), device)
        b_ppsip[0] = pp

        if pp >= thr:
            b_class[0] = CLASS_PSIP
            # keep others at 0.0
            tout.Fill()
            n_written += 1
            continue

        # (3) DY vs comb: outputs p(DY) (per training POS=DY, NEG=COMB)
        pdy = prob_pos(model_dy, apply_scaler(feats, scaler_dy), device)
        b_pdy[0] = pdy
        b_pcomb[0] = 1.0 - pdy

        b_class[0] = CLASS_DY if pdy >= thr else CLASS_COMB
        tout.Fill()
        n_written += 1

        if (i % 20000) == 0 and i > 0:
            print(f"[INFO] processed {i}/{n_in}  written={n_written}")

    # ---- write output ----
    tout.Write()
    fout.Close()
    fin.Close()

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "tree": args.tree,
        "thr": thr,
        "mass_min": args.mass_min,
        "mass_max": args.mass_max,
        "n_entries_input": n_in,
        "n_entries_pass_mass": kept_mass,
        "n_entries_skipped_zero_vectors": skipped_zero,
        "n_entries_written": n_written,
        "class_encoding": {"1": "jpsi", "2": "psip", "3": "dy", "4": "comb"},
        "checkpoints": {
            "jpsi": str(args.ckpt_jpsi),
            "psip": str(args.ckpt_psip),
            "dy": str(args.ckpt_dy),
        },
    }
    print(json.dumps(summary, indent=2))
    print("[DONE] Wrote tagged ROOT file with branches: ml_class, ml_p_jpsi, ml_p_psip, ml_p_dy, ml_p_comb")


if __name__ == "__main__":
    main()

