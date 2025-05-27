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

## Examples

| What we want to run | Command |
|---------------------|---------|
| **1 MPI rank** (serial run), 0.75 window overlap, **1 walker/window**, 500 sweeps/iter, RNG seed = 42, *L = 4*, *q = 2* |<br>`mpiexec -n 1 python rewl_multi.py 0.75 1 500 42 4 2 1.0001 0.8` |
| **3 MPI ranks**, 0.50 overlap, **1 walker/window**, 1000 sweeps/iter, seed = 14 *(matches the original C++ example)* |<br>`mpiexec -n 3 python rewl_multi.py 0.5 1 1000 14 4 2 1.0001 0.8` |
| **8 MPI ranks**, 0.75 overlap, **2 walkers/window** (⇒ 4 windows), 2000 sweeps/iter, seed = 314, *L = 10*, *q = 10* |<br>`mpiexec -n 8 python rewl_multi.py 0.75 2 2000 314 10 10 1.000001 0.8` |

* `overlap`  – fraction (0–1) of energy range shared between neighbouring windows  
* `walkers/window` – number of independent WL walkers in each window  
* `sweeps/iter` – Metropolis sweeps between histogram-flatness checks  
* `seed`    – random-number seed offset (ranks add 17 × rank)  
* `L`, `q`   – lattice size and Potts-state count  
* `f_final`  – stop when ln *f* ≤ ln (*f_final*)  
* `flatness`  – histogram flatness criterion (e.g. 0.8 = 80 %)


# 4. stitch density of states + plots
python analyse_rewl.py
# --> ln_g_stitched.dat, lng_curve.png, beta_curve.png, caloric_curve.png

