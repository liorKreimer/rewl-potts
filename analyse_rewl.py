#!/usr/bin/env python3
"""
analyse_rewl.py  –  stitch, smooth β(E), and plot caloric curve
===============================================================

• Reads ln_g_rank<ID>.dat slices (2 columns:  E  ln_g).
• Stitches & rescales exactly as before.
• Drops bins where ln g == 0.
• Smooths β(E) with a 5-point Savitzky–Golay filter.
• Saves 3 plots (PNG) and ln_g_stitched.dat.

Run inside the directory that contains ln_g_rank*.dat .
"""

import glob, sys, numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from scipy.signal import savgol_filter
except ImportError:
    sys.exit("scipy not found.  Install it with  pip install scipy  and rerun.")

# --------------------------------------------------------------------------- helpers
def load_slices():
    files = sorted(glob.glob("ln_g_rank*.dat"))
    if not files:
        sys.exit("No ln_g_rank*.dat files found – run REWL first.")
    arrays = [np.loadtxt(f) for f in files]
    E = arrays[0][:, 0]
    ln_stack = np.vstack([a[:, 1] for a in arrays])
    return E, ln_stack

def derivative(y, x):
    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    dy[0] = dy[1]; dy[-1] = dy[-2]
    return dy

def stitch_rescale(E, ln_stack):
    # max over ranks but keep zeros intact
    ln_max = np.where(ln_stack.any(axis=0), ln_stack.max(axis=0), 0.0)
    stitched = ln_stack[0].copy()
    for i in range(1, ln_stack.shape[0]):
        piece = ln_stack[i]
        mask = (stitched != 0) & (piece != 0)
        if not np.any(mask):
            stitched = np.where(piece != 0, piece, stitched)
            continue
        beta_s = derivative(stitched, E)
        beta_p = derivative(piece,    E)
        join   = np.where(mask, np.abs(beta_s - beta_p), np.inf).argmin()
        shift  = stitched[join] - piece[join]
        stitched = np.where(piece != 0, piece + shift, stitched)
    # normalise: ln g(E_min) = 0  (ignore zeros)
    nz = stitched != 0
    stitched = np.where(nz, stitched - stitched[nz][0], 0.0)
    return stitched

# --------------------------------------------------------------------------- main
def main():
    E, ln_stack = load_slices()
    ln_full = stitch_rescale(E, ln_stack)

    # drop zero bins
    mask = ln_full != 0
    E_nz  = E[mask]
    ln_nz = ln_full[mask]

    # β(E) raw & smoothed
    beta_raw = derivative(ln_full, E)[mask]
    # ensure we have at least 5 points for Savitzky–Golay
    window = 5 if len(beta_raw) >= 5 else len(beta_raw) | 1
    beta_smooth = savgol_filter(beta_raw, window_length=window, polyorder=2)

    # caloric curve: T(E) = 1 / β(E)  (take only β>0 to avoid sign flips)
    pos = beta_smooth > 0
    T_micro = 1.0 / beta_smooth[pos]
    E_T     = E_nz[pos]

    # ------------- save stitched DOS -----------------
    np.savetxt("ln_g_stitched.dat",
               np.column_stack((E, ln_full)),
               header="E   ln_g(E) – stitched, rescaled, normalised")

    # ------------- plot ln g(E) ----------------------
    plt.figure(figsize=(5, 3.8))
    plt.plot(E_nz, ln_nz, lw=1)
    plt.xlabel("Energy E"); plt.ylabel("ln g(E)")
    plt.title("Stitched ln g(E)")
    plt.tight_layout()
    out1 = Path("lng_curve.png"); plt.savefig(out1, dpi=150); plt.close()

    # ------------- plot β(E) -------------------------
    plt.figure(figsize=(5, 3.8))
    plt.plot(E_nz, beta_smooth, lw=1)
    plt.xlabel("Energy E"); plt.ylabel("β(E) = d ln g / dE")
    plt.title("Micro-canonical β(E)")
    plt.tight_layout()
    out2 = Path("beta_curve.png"); plt.savefig(out2, dpi=150); plt.close()

    # ------------- plot caloric curve T(E) -----------
    plt.figure(figsize=(5, 3.8))
    plt.plot(E_T, T_micro, lw=1)
    plt.xlabel("Energy E"); plt.ylabel("T(E) = 1 / β(E)")
    plt.title("Micro-canonical caloric curve")
    plt.tight_layout()
    out3 = Path("caloric_curve.png"); plt.savefig(out3, dpi=150); plt.close()

    # summary
    print(f"✓  ln_g_stitched.dat written  ({E.size} rows)")
    print(f"✓  Plots saved:\n   • {out1.resolve()}\n   • {out2.resolve()}\n   • {out3.resolve()}")

if __name__ == "__main__":
    main()
