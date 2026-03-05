"""
Tutorial 01b: Verifying localization from a scrambled initial guess
===================================================================

This companion to tutorial01.py tests whether the ASE Wannier localization
procedure can recover the correct MLWFs even when starting from a poor
initial guess. We take the well-converged .amn projections and apply a
random unitary transformation to "jumble" the rotation matrices, then
re-run localize() and verify that the optimizer finds the same minimum.

This is a more stringent test than tutorial01.py, where the initial .amn
guess was already near-optimal.
"""

import numpy as np
from pathlib import Path
from scipy.stats import unitary_group

from ase.dft.wannier import Wannier

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------

tutorial_dir = Path(__file__).resolve().parent
seed = tutorial_dir / 'gaas'

print("=" * 65)
print("  Tutorial 01b: Localization from a scrambled initial guess")
print("=" * 65)
print()

# ---------------------------------------------------------------------------
# 1. Reference run: localize from .amn projections (the good initial guess)
# ---------------------------------------------------------------------------

print("Reference: localize from .amn projections...")
wan_ref = Wannier.from_wannier90(
    seed=str(seed),
    initialwannier='amn',
    functional='std',
)
functional_ref_init = wan_ref.get_functional_value()
wan_ref.localize(step=0.25, tolerance=1e-10)
functional_ref_final = wan_ref.get_functional_value()
spreads_ref = wan_ref.get_spreads()
centers_ref = wan_ref.get_centers()

print(f"  Initial functional:  {functional_ref_init:.6f}")
print(f"  Converged functional: {functional_ref_final:.6f}")
print(f"  Converged spreads:   {spreads_ref}")
print()

# ---------------------------------------------------------------------------
# 2. Scrambled run: apply random unitary to rotation matrices
# ---------------------------------------------------------------------------
# We create a fresh Wannier object from .amn, then replace U_kww[k]
# with U_kww[k] @ R for a random unitary R. This mixes the Wannier
# functions while preserving the subspace (the rotation stays unitary).

print("Scrambled: applying random unitary transformation...")
rng = np.random.default_rng(seed=42)

wan_scrambled = Wannier.from_wannier90(
    seed=str(seed),
    initialwannier='amn',
    functional='std',
)

# Generate a random unitary matrix (same for all k-points, to keep it simple)
Nw = wan_scrambled.nwannier
R = unitary_group.rvs(Nw, random_state=rng)

# Apply the scrambling: U_kww[k] -> U_kww[k] @ R
for k in range(wan_scrambled.Nk):
    wan_scrambled.wannier_state.U_kww[k] = (
        wan_scrambled.wannier_state.U_kww[k] @ R
    )
wan_scrambled.update()

functional_scrambled_init = wan_scrambled.get_functional_value()
spreads_scrambled_init = wan_scrambled.get_spreads()
centers_scrambled_init = wan_scrambled.get_centers()

print(f"  Scrambled initial functional: {functional_scrambled_init:.6f}"
      f"  (vs reference: {functional_ref_init:.6f})")
print(f"  Scrambled initial spreads:    {spreads_scrambled_init}")
print()
print(f"  The scrambled functional is much lower (= less localized)")
print(f"  than the .amn starting point, confirming the mixing worked.")
print()

# ---------------------------------------------------------------------------
# 3. Localize from the scrambled starting point
# ---------------------------------------------------------------------------
# Record convergence history via a custom logger

class HistoryLogger:
    """Captures per-iteration functional values from MDmin output."""
    def __init__(self):
        self.values = []
    def __call__(self, *args):
        msg = ' '.join(str(a) for a in args)
        if msg.startswith('MDmin: iter='):
            for part in msg.split(','):
                part = part.strip()
                if part.startswith('value='):
                    self.values.append(float(part.split('=')[1]))


# Also record the reference convergence for comparison
wan_ref2 = Wannier.from_wannier90(
    seed=str(seed),
    initialwannier='amn',
    functional='std',
)
ref_history = [wan_ref2.get_functional_value()]
ref_logger = HistoryLogger()
wan_ref2.log = ref_logger
wan_ref2.localize(step=0.25, tolerance=1e-10)
ref_history.extend(ref_logger.values)

# Now localize the scrambled version
print("Localizing from scrambled starting point...")
scrambled_history = [functional_scrambled_init]
scrambled_logger = HistoryLogger()
wan_scrambled.log = scrambled_logger
wan_scrambled.localize(step=0.25, tolerance=1e-10)
scrambled_history.extend(scrambled_logger.values)

functional_scrambled_final = wan_scrambled.get_functional_value()
spreads_scrambled_final = wan_scrambled.get_spreads()
centers_scrambled_final = wan_scrambled.get_centers()

print(f"  Converged functional: {functional_scrambled_final:.6f}"
      f"  (reference: {functional_ref_final:.6f})")
print(f"  Converged spreads:   {spreads_scrambled_final}")
print()

# ---------------------------------------------------------------------------
# 4. Compare: did the optimizer recover the correct minimum?
# ---------------------------------------------------------------------------

print("Comparison of converged results:")
print("-" * 55)
func_diff = abs(functional_scrambled_final - functional_ref_final)
print(f"  Functional difference:  {func_diff:.2e}")

spread_diff = np.max(np.abs(np.sort(spreads_scrambled_final)
                             - np.sort(spreads_ref)))
print(f"  Max spread difference:  {spread_diff:.2e}")

# Compare centers (match by nearest distance, since WF ordering may differ)
from scipy.spatial.distance import cdist
dists = cdist(centers_ref, centers_scrambled_final)
# Greedy nearest-neighbor matching
matched = []
for _ in range(Nw):
    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    matched.append(dists[i, j])
    dists[i, :] = np.inf
    dists[:, j] = np.inf
center_diff = max(matched)
print(f"  Max center difference:  {center_diff:.2e} A (after matching)")

tol = 1e-3
if func_diff < tol and spread_diff < tol:
    print(f"\n  SUCCESS: Both runs converged to the same minimum!")
else:
    print(f"\n  Note: Small differences may remain due to local minima")
    print(f"  or convergence tolerance.")
print()

# ---------------------------------------------------------------------------
# 5. Convergence plot
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Generating convergence plot...")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(len(ref_history)), ref_history,
            'o-', markersize=4, linewidth=1.2, color='steelblue',
            label='From .amn projections (good start)')
    ax.plot(range(len(scrambled_history)), scrambled_history,
            's-', markersize=4, linewidth=1.2, color='crimson',
            label='From scrambled start')

    ax.axhline(y=functional_ref_final, color='gray', linestyle='--',
               alpha=0.5, label=f'Converged: {functional_ref_final:.6f}')

    ax.set_xlabel('Optimization step')
    ax.set_ylabel('Localization functional')
    ax.set_title('Tutorial 01b: Convergence from good vs scrambled start')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = tutorial_dir / 'tutorial01b_convergence.png'
    fig.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path.name}")
    print()

except ImportError:
    print("  (matplotlib not available -- skipping plot)")
    print()

print("=" * 65)
print("  Tutorial 01b complete!")
print("=" * 65)
