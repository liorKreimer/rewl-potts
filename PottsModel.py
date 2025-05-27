# potts_model.py
"""
Pure-Potts implementation translated line-by-line from WLpotts_mpi.cpp
(T. Vogel & Y.-W. Li, 2013-19).

Only model-specific responsibilities live here:

    • building the L×L lattice
    • periodic-boundary neighbour bookkeeping
    • local & total energy evaluation
    • proposing a single-spin update and returning ΔE

All parallel / WL control logic is intentionally absent.
"""

from __future__ import annotations
import numpy as np
import random
from dataclasses import dataclass, field


@dataclass
class PottsModel:
    L: int = 4                # linear dimension (L1dim in C++)
    q: int = 10               # number of spin states
    bc: str = "periodic"      # 'periodic' or 'braskamp_kunz'
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self.N = self.L ** 2                       # total number of spins
        self._build_neighbour_table()              # neighbour indices
        self.lattice = np.zeros(self.N, dtype=int) # spins live in flat array
        self.init_lattice_random()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _build_neighbour_table(self):
        """Pre-compute 4 neighbours (up, right, down, left) for every site
        with the chosen boundary condition."""
        nbr = np.empty((self.N, 4), dtype=int)

        for idx in range(self.N):
            row, col = divmod(idx, self.L)
            up        = (row - 1) % self.L,  col
            right     =  row, (col + 1) % self.L
            down      = (row + 1) % self.L,  col
            left      =  row, (col - 1) % self.L
            nbr[idx]  = np.ravel_multi_index(
                np.array([up, right, down, left]).T, (self.L, self.L)
            )

        # Optional Braskamp–Kunz BC tweaks (same as C++ bctype==1)
        if self.bc == "braskamp_kunz":
            top_fixed         = np.ravel_multi_index(
                ([0] * self.L, np.arange(self.L)), (self.L, self.L)
            )
            bottom_fixed_even = np.ravel_multi_index(
                ([self.L - 1] * self.L, np.arange(self.L)), (self.L, self.L)
            )
            nbr[top_fixed, 0]   = self.N        # ghost spin +1
            nbr[bottom_fixed_even, 2] = self.N + 1  # ghost spin –1

        self.neighbour = nbr

    def init_lattice_random(self):
        """Fill lattice with random spin orientations (uniform in [0,q-1])."""
        self.lattice[:] = self.rng.choices(range(self.q), k=self.N)

    def init_lattice_equal(self, value: int = 0):
        """Fill lattice homogeneously with the given spin value."""
        if not (0 <= value < self.q):
            raise ValueError("spin value out of range")
        self.lattice.fill(value)

    # ------------------------------------------------------------------
    # Energy routines – faithful to C++ implementation
    # ------------------------------------------------------------------
    def _local_energy(self, site: int) -> int:
        """Energy contribution of one spin (counts matching neighbours)."""
        same = self.lattice[site] == self.lattice[self.neighbour[site]]
        return -int(np.sum(same))                 # one unit per satisfied bond
        # C++ version: local_energy() returns negative count :contentReference[oaicite:0]{index=0}

    def total_energy(self) -> int:
        """Total Potts Hamiltonian with double-count avoidance.

        For periodic BC we sum matches to 'up' and 'right' neighbours only,
        exactly as done in C++ totalenergy() :contentReference[oaicite:1]{index=1}.
        """
        e = 0
        for site in range(self.N):
            # up (neighbour[site,0]) and right (neighbour[site,1])
            for neigh in self.neighbour[site, :2]:
                if self.lattice[site] == self.lattice[neigh]:
                    e -= 1
        if self.bc == "braskamp_kunz":
            # Braskamp–Kunz adds fixed upper boundaries (matches to 'up')
            for col in range(self.L):
                idx = col                                  # top row index
                if self.lattice[idx] == self.lattice[self.neighbour[idx, 0]]:
                    e -= 1
        return e

    # ------------------------------------------------------------------
    # Monte-Carlo move
    # ------------------------------------------------------------------
    def propose_update(self, site: int | None = None) -> int:
        """
        Change a single spin to a random new state in [0,q-1] (≠ old by chance)
        and **return the energy difference ΔE = E_new – E_old**.

        Side effect: the lattice IS modified here, mirroring the C++ behaviour
        (caller must revert on reject). :contentReference[oaicite:2]{index=2}
        """
        if site is None:
            site = self.rng.randrange(self.N)

        e_before = self._local_energy(site)
        old_spin = self.lattice[site]

        # draw until different from current to avoid ΔE = 0 self-updates
        new_spin = self.rng.randrange(self.q)
        while new_spin == old_spin:
            new_spin = self.rng.randrange(self.q)
        self.lattice[site] = new_spin

        e_after = self._local_energy(site)
        return e_after - e_before, old_spin, new_spin, site

    # ------------------------------------------------------------------
    # Global Potts constants handy for WL bookkeeping
    # ------------------------------------------------------------------
    @property
    def energy_bounds(self) -> tuple[int, int]:
        """Return (E_min, E_max) for 2-D square lattice Potts with this L."""
        E_min = -2 * self.N               # derived in C++ as Eglobalmin :contentReference[oaicite:3]{index=3}
        E_max = 0
        return E_min, E_max

'''
# ---------------------- Test Suite ----------------------
def test_energy_consistency(L=6, q=8, moves=5000, seed=123):
    rng = random.Random(seed)
    potts = PottsModel(L=L, q=q, rng=rng)
    E = potts.total_energy()

    for _ in range(moves):
        dE, old, new, site = potts.propose_update()
        E2 = potts.total_energy()
        assert E2 - E == dE, f"Energy mismatch: ΔE={dE}, recomputed={E2 - E}"
        # revert half the time to stress both branches
        if rng.random() < 0.5:
            potts.lattice[site] = old
            E = potts.total_energy()
        else:
            E = E2
    return True

def test_energy_bounds(L=4, q=10):
    potts = PottsModel(L=L, q=q)
    Emin, Emax = potts.energy_bounds
    assert Emin == -2 * potts.N and Emax == 0, "Energy bounds incorrect"
    return True


if __name__ == "__main__":
    assert test_energy_consistency()
    assert test_energy_bounds()

    print("All PottsModel sanity checks passed ✔️")
# ---------------------- End of PottsModel.py ----------------------
'''

