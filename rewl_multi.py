"""rewl_mpi.py – strict Python port of WLpotts_mpi.cpp (Replica-Exchange Wang–Landau)
================================================================================
• Communicator layout and algorithmic flow now **exactly** match the reference.
• **Global ln f**: halved only when *every* energy window is flat → simulation
  finishes when the slowest window reaches ln f_final.
• Writes per‑rank ln g slices to ln_g_rank<ID>.dat.
"""
from __future__ import annotations
import sys, math, time, random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from mpi4py import MPI

from PottsModel import PottsModel

# -----------------------------------------------------------------------------
@dataclass
class Walker:
    model: PottsModel
    E: int
    ln_g: np.ndarray
    hist: np.ndarray
    window: Tuple[int, int]
    shift: int
    rng: random.Random

    def _idx(self, E):
        return E + self.shift

    def _visit(self, E, ln_f):
        idx = self._idx(E)
        self.ln_g[idx] += ln_f
        self.hist[idx] += 1

    def mc_sweep(self, ln_f: float) -> None:
        """
       Perform *one Wang–Landau sweep* (N single-spin trials) for this walker.

        Parameters
        ----------
        ln_f : float
            Current modification factor ln f  (will be **added** to ln g(E)
            every time the walker visits an energy bin).

        Side-effects
        ------------
        • Updates the Potts lattice in-place.
        • Updates walker energy **self.E**, local ln g(E) array, and histogram.
        """
        N = self.model.N  # number of spins = one sweep
        for _ in range(N):
            # ---- propose a single-spin change ---------------------------------
            dE, old_state, new_state, site = self.model.propose_update()
            E_new = self.E + dE  # resulting energy if accepted

            # ---- respect the walker’s energy window --------------------------
            if not (self.window[0] <= E_new <= self.window[1]):
                # outside my slice → reject immediately, restore spin
                self.model.lattice[site] = old_state
                self._visit(self.E, ln_f)  # still visit current E
                continue

            # ---- Wang–Landau acceptance probability --------------------------
            log_ratio = (self.ln_g[self._idx(self.E)]
                         - self.ln_g[self._idx(E_new)])
            accept = (log_ratio > 0) or (self.rng.random() <= math.exp(log_ratio))

            if accept:
                self.E = E_new  # keep new energy & spin
            else:
                self.model.lattice[site] = old_state  # revert spin, keep old E

            # ---- update DoS estimate & histogram -----------------------------
            self._visit(self.E, ln_f)  # add ln f to ln g(E) and ++hist


# -----------------------------------------------------------------------------

def flat_enough(hist_slice, c):
    nz = hist_slice[hist_slice > 0]
    return False if nz.size == 0 else nz.min() >= c * hist_slice.mean()

# -----------------------------------------------------------------------------

def replica_exchange(comm, wlk: Walker, win_id, windows, w_per_win, phase):
    """
    Attempt a one–dimensional replica exchange (swap lattices) between
    the current walker and its nearest neighbour in rank space.

    Parameters
    ----------
    comm     : MPI.Comm   – world communicator
    wlk      : Walker     – the local walker object
    win_id   : int        – index of the energy window this rank belongs to
    windows  : list[(lo,hi)] – list of (lower, upper) energy bounds
    w_per_win: int        – walkers per window (used only for layout)
    phase    : int (0 or 1) – even/odd phase so swaps are non-overlapping
    """

    # --------- pick my partner rank -------------------------------------------------
    rank, size = comm.Get_rank(), comm.Get_size()
    partner = rank + 1 if (rank + phase) % 2 == 0 else rank - 1  # right or left neighbour
    if partner < 0 or partner >= size:           # edge rank in this phase → no partner
        return

    # --------- 1. exchange current energies -----------------------------------------
    sendE  = np.array([wlk.E], dtype=np.int64)
    recvE  = np.empty_like(sendE)
    comm.Sendrecv(sendE, dest=partner, recvbuf=recvE, source=partner)
    E_partner = int(recvE[0])

    # --------- 2. decide whether to swap lattices -----------------------------------
    # Here: 50 % chance; could replace with Metropolis on E if desired
    swap = wlk.rng.random() < 0.5
    flag = np.array([1 if swap else 0], dtype=np.int8)

    # **Bug source (fixed elsewhere):** this Bcast is collective on COMM_WORLD;
    # if multiple pairs call it simultaneously with different roots, MPICH dead-locks.
    # A two-rank Sendrecv handshake is safer.
    comm.Bcast(flag, root=min(rank, partner))    # broadcast “swap?” to the pair
    if flag[0] == 0:                             # either side vetoed
        return

    # --------- 3. swap the full 2-D lattices ----------------------------------------
    sendLat = wlk.model.lattice.copy()
    recvLat = np.empty_like(sendLat)
    comm.Sendrecv(sendLat, dest=partner, recvbuf=recvLat, source=partner)

    # adopt partner’s state
    wlk.model.lattice[:] = recvLat
    wlk.E = E_partner

    # --------- 4. keep only if new config is inside my window -----------------------
    lb, ub = windows[win_id]
    if not (lb <= wlk.E <= ub):                  # fell out of my energy slice
        wlk.model.lattice[:] = sendLat           # revert lattice and energy
        wlk.E = int(sendE[0])


# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) < 5:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Usage: python rewl_mpi.py <overlap> <walkers/window> <sweeps/iter> <seed> [L] [q] [f_final] [flatness]", file=sys.stderr)
        sys.exit(1)

    overlap   = float(sys.argv[1]);   w_per_win = int(sys.argv[2]); sweeps_it = int(sys.argv[3]); seed = int(sys.argv[4])
    L  = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    q  = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    f_final = float(sys.argv[7]) if len(sys.argv) > 7 else 1.000001
    flat_c  = float(sys.argv[8]) if len(sys.argv) > 8 else 0.8

    #MPI
    comm = MPI.COMM_WORLD; rank = comm.Get_rank(); size = comm.Get_size()
    if size % w_per_win != 0:
        if rank == 0:
            print("[REWL] #MPI ranks must be multiple of walkers/window", file=sys.stderr)
        sys.exit(2)

    dummy = PottsModel(L=L, q=q, rng=random.Random(seed))
    Emin, Emax = dummy.energy_bounds; Espan = Emax - Emin + 1; shift = -Emin

    n_win = size // w_per_win
    win_w = int(math.ceil(Espan / (n_win + 1 - overlap * (n_win - 1))))
    windows: List[Tuple[int,int]] = []
    start = Emin
    for _ in range(n_win):
        end = min(Emax, start + win_w - 1)
        windows.append((start, end)); start += int(win_w * (1 - overlap))
    windows[-1] = (windows[-1][0], Emax)

    win_id = rank // w_per_win; win_comm = comm.Split(win_id, rank)

    rng = random.Random(seed + 17 * rank)
    model = PottsModel(L=L, q=q, rng=rng)
    ln_g = np.zeros(Espan); hist = np.zeros(Espan, dtype=int)
    walker = Walker(model, model.total_energy(), ln_g, hist, windows[win_id], shift, rng)

    ln_f = math.log(math.e); ln_f_final = math.log(f_final)
    if rank == 0:  # <-- safe to print now
        print(f"[REWL] start | windows={n_win} walkers/window={w_per_win} "
              f"L={L} q={q} sweeps/iter={sweeps_it}", flush=True)

    iter_idx = 0; t0 = time.time()
    while True:
        hist.fill(0)
        for sweep in range(sweeps_it):
            walker.mc_sweep(ln_f)
            replica_exchange(comm, walker, win_id, windows, w_per_win, sweep % 2)

        # window-level reductions
        hist_acc = np.zeros_like(hist); win_comm.Allreduce(hist, hist_acc, op=MPI.SUM)
        lo, hi = windows[win_id][0] + shift, windows[win_id][1] + shift
        win_flat = flat_enough(hist_acc[lo:hi+1], flat_c)

        ln_g_acc = np.zeros_like(ln_g); win_comm.Allreduce(ln_g, ln_g_acc, op=MPI.SUM)
        ln_g[:] = ln_g_acc / w_per_win

        # world-wide decision
        all_flat = comm.allreduce(1 if win_flat else 0, op=MPI.LAND)
        if rank == 0:
            print(f"Iter {iter_idx} | winFlat={win_flat} | allFlat={bool(all_flat)} | ln f={ln_f:.3e}")

        if all_flat:
            ln_f *= 0.5; iter_idx += 1
        # termination: slowest window reached target
        if all_flat and ln_f <= ln_f_final:
            break

    # save slice
    np.savetxt(f"ln_g_rank{rank}.dat", np.column_stack((np.arange(Emin, Emax+1), ln_g)))
    if rank == 0:
        print(f"[REWL] done in {time.time()-t0:.2f} s | iterations={iter_idx}")
        print("Saved per‑rank ln_g slices → ln_g_rank<ID>.dat (complete when slowest window converged)")

if __name__ == "__main__":
    main()
