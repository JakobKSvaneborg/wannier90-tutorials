"""
Tutorial 02b: Verifying localization from a scrambled initial guess (Lead)
=========================================================================

Companion to tutorial 02. We take the .amn projections for FCC Lead (Pb)
with 4 sp3 Wannier functions and apply a k-dependent random unitary
transformation, then verify that the optimizer recovers the same minimum.
"""

import numpy as np
from pathlib import Path
from scipy.linalg import expm

from ezwannier import Wannier

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------

tutorial_dir = Path(__file__).resolve().parent
seed = tutorial_dir / 'lead'

print("=" * 65)
print("  Tutorial 02b: Localization from a scrambled initial guess")
print("  System: Lead (Pb), 4 sp3 Wannier functions")
print("=" * 65)
print()

# ---------------------------------------------------------------------------
# 1. Reference run: localize from .amn projections
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
# 2. Scrambled run: apply k-dependent random unitary
# ---------------------------------------------------------------------------
# R(k) = expm(i * H(k)) where H(k) is a smoothly k-varying Hermitian
# matrix built from cos(2*pi*k.n + phi) basis functions.

print("Scrambled: applying k-dependent random unitary transformation...")
rng = np.random.default_rng(seed=42)

wan_scrambled = Wannier.from_wannier90(
    seed=str(seed),
    initialwannier='amn',
    functional='std',
)

Nw = wan_scrambled.nwannier
Nk = wan_scrambled.Nk
kpts = -wan_scrambled.kpt_kc  # un-negate to get original W90 k-points

# Build a basis of Hermitian matrices for the Lie algebra of U(Nw)
basis = []
for i in range(Nw):
    H = np.zeros((Nw, Nw), complex)
    H[i, i] = 1.0
    basis.append(H)
for i in range(Nw):
    for j in range(i + 1, Nw):
        H_sym = np.zeros((Nw, Nw), complex)
        H_sym[i, j] = 1.0
        H_sym[j, i] = 1.0
        basis.append(H_sym)
        H_anti = np.zeros((Nw, Nw), complex)
        H_anti[i, j] = 1j
        H_anti[j, i] = -1j
        basis.append(H_anti)

n_terms = len(basis)
directions = rng.integers(-1, 2, size=(n_terms, 3))
amplitudes = rng.uniform(-1, 1, size=n_terms)
phases = rng.uniform(0, 2 * np.pi, size=n_terms)
strength = 1.5

for k in range(Nk):
    H_k = np.zeros((Nw, Nw), complex)
    for j in range(n_terms):
        f = np.cos(2 * np.pi * np.dot(kpts[k], directions[j]) + phases[j])
        H_k += strength * amplitudes[j] * f * basis[j]
    R_k = expm(1j * H_k)
    wan_scrambled.wannier_state.U_kww[k] = (
        wan_scrambled.wannier_state.U_kww[k] @ R_k
    )

wan_scrambled.update()

functional_scrambled_init = wan_scrambled.get_functional_value()
spreads_scrambled_init = wan_scrambled.get_spreads()

print(f"  Scrambled initial functional: {functional_scrambled_init:.6f}"
      f"  (vs reference: {functional_ref_init:.6f})")
print(f"  Scrambled initial spreads:    {spreads_scrambled_init}")
print()

# ---------------------------------------------------------------------------
# 3. Localize from the scrambled starting point
# ---------------------------------------------------------------------------

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


# Reference convergence history
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

# Scrambled convergence history
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
# 4. Compare results
# ---------------------------------------------------------------------------

print("Comparison of converged results:")
print("-" * 55)
func_diff = abs(functional_scrambled_final - functional_ref_final)
print(f"  Functional difference:  {func_diff:.2e}")

spread_diff = np.max(np.abs(np.sort(spreads_scrambled_final)
                             - np.sort(spreads_ref)))
print(f"  Max spread difference:  {spread_diff:.2e}")

print()
print(f"  Ref centers (Angstrom):")
for w in range(len(centers_ref)):
    c = centers_ref[w]
    print(f"    WF {w+1}: ({c[0]:7.4f}, {c[1]:7.4f}, {c[2]:7.4f})")
print(f"  Scrambled centers (Angstrom):")
for w in range(len(centers_scrambled_final)):
    c = centers_scrambled_final[w]
    print(f"    WF {w+1}: ({c[0]:7.4f}, {c[1]:7.4f}, {c[2]:7.4f})")
print()
print("  Note: Centers may differ by lattice translations -- the")
print("  k-dependent scrambling can shift WFs to equivalent bonds.")

tol = 1e-3
if func_diff < tol and spread_diff < tol:
    print(f"\n  SUCCESS: Both runs converged to the same minimum!")
    print(f"  (functional and spreads match to within {tol})")
else:
    print(f"\n  Note: Differences remain -- may indicate local minima")
    print(f"  or insufficient convergence.")
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
    ax.set_title('Tutorial 02b: Lead — Convergence from good vs scrambled start')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = tutorial_dir / 'tutorial02b_convergence.png'
    fig.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path.name}")
    print()

except ImportError:
    print("  (matplotlib not available -- skipping plot)")
    print()

print("=" * 65)
print("  Tutorial 02b complete!")
print("=" * 65)
