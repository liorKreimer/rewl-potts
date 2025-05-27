# Replica-Exchange Wang–Landau sampler for the *q*-state Potts model

Python / MPI re-implementation of the REWL algorithm  
based on Ying-Wai Li *et al.* **BMSP 2017** (GitHub: `BMSP2017`).

| File | Purpose |
|------|---------|
| `PottsModel.py`   | Stand-alone *q*-state Potts model (lattice, energy, single-spin proposal). |
| `rewl_multi.py`   | MPI driver – multiple windows, multiple walkers per window, replica exchange. |
| `analyse_rewl.py` | Offline stitch & normalise `ln g(E)`, smooth β(E), plot PNGs. |

---

## Quick start

```bash
# 1. clone the repo
git clone https://github.com/liorKreimer/rewl-potts.git
cd rewl-potts

# 2. create / activate a conda env with MPI support
conda create -n mpi_env python=3.10 numpy scipy matplotlib mpich mpi4py
conda activate mpi_env

# 3. run a small demo (8 ranks, 2 walkers/window, L=10, q=10)
mpiexec -n 8 python rewl_multi.py 0.75 2 200 314 10 10 1.01 0.8

# 4. stitch density of states + plots
python analyse_rewl.py
# --> ln_g_stitched.dat, lng_curve.png, beta_curve.png, caloric_curve.png
