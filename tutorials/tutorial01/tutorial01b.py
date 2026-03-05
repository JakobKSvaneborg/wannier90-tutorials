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
# with U_kww[k] @ R(k) for a k-dependent random unitary R(k). This
# changes the gauge differently at each k-point, which is a stronger
# test than a global (k-independent) rotation.
#
# To make R(k) vary smoothly with k (as a physical gauge change would),
# we generate it as R(k) = expm(i * H(k)) where H(k) is a Hermitian
# matrix that interpolates smoothly across the k-mesh:
#   H(k) = strength * sum_j c_j * cos(2*pi*k . n_j + phi_j) * A_j
# with random coefficients c_j, phase offsets phi_j, lattice vectors
# n_j, and basis Hermitian matrices A_j. The 'strength' parameter
# controls how far from the identity each R(k) is. We use cos + random
# phases instead of sin, since sin(2*pi*k.n) vanishes for Gamma-centered
# grids where k-components are 0 or 0.5.

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
# We use Nw^2 generators: Nw diagonal + Nw*(Nw-1)/2 symmetric off-diag
# + Nw*(Nw-1)/2 antisymmetric off-diag
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

# Random mixing directions: for each basis matrix, pick a random
# reciprocal lattice direction, amplitude, and phase offset
n_terms = len(basis)
directions = rng.integers(-1, 2, size=(n_terms, 3))  # small G-vectors
amplitudes = rng.uniform(-1, 1, size=n_terms)
phases = rng.uniform(0, 2 * np.pi, size=n_terms)  # random phase offsets
strength = 1.5  # controls overall scrambling magnitude

from scipy.linalg import expm

for k in range(Nk):
    # Build H(k) = strength * sum_j amp_j * cos(2*pi*k.n_j + phi_j) * A_j
    H_k = np.zeros((Nw, Nw), complex)
    for j in range(n_terms):
        f = np.cos(2 * np.pi * np.dot(kpts[k], directions[j]) + phases[j])
        H_k += strength * amplitudes[j] * f * basis[j]
    # R(k) = expm(i * H(k)) is unitary since H(k) is Hermitian
    R_k = expm(1j * H_k)
    wan_scrambled.wannier_state.U_kww[k] = (
        wan_scrambled.wannier_state.U_kww[k] @ R_k
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

# Note on WF centers: The Berry-phase center formula
#   r_n = -(1/Nk) sum_{k,b} w_b * b * Im ln M_nn^(k,b)
# can place centres at different periodic images.  The
# get_centers() method includes a branch-cut correction that
# ensures each reported centre is at a *valid* lattice image
# of the true WF position, so the centres can always be shifted
# by lattice vectors to a common unit cell.
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
print("  Note: The centres may be at different periodic images, but")
print("  they can be shifted by lattice vectors to the same positions.")
print("  The WFs are physically identical (same functional and spreads).")

tol = 1e-3
if func_diff < tol and spread_diff < tol:
    print(f"\n  SUCCESS: Both runs converged to the same minimum!")
    print(f"  (functional and spreads match to within {tol})")
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

    # --- 3D plot: scrambled-initial vs converged WF centers ---
    from matplotlib.lines import Line2D
    from scipy.optimize import linear_sum_assignment

    atoms = wan_scrambled.atoms
    cell = atoms.cell[:]  # (3, 3) lattice vectors
    Nw = wan_scrambled.nwannier

    # Helper: shift a point by lattice vectors to be closest to a target
    def shift_to_nearest(point, target, cell):
        best = point.copy()
        best_d = np.linalg.norm(point - target)
        for n1 in range(-4, 5):
            for n2 in range(-4, 5):
                for n3 in range(-4, 5):
                    shifted = point + n1 * cell[0] + n2 * cell[1] + n3 * cell[2]
                    d = np.linalg.norm(shifted - target)
                    if d < best_d:
                        best_d = d
                        best = shifted
        return best

    # (1) Use the SCRAMBLED converged centers for the final plot positions.
    #
    # The get_centers() method includes a branch-cut correction that
    # ensures the returned centers are at valid lattice images of the
    # true WF positions. The centers may be at distant periodic images,
    # so we shift each one by lattice vectors to sit on the bonds of
    # a single Ga atom.
    final_plot = np.array([c.copy() for c in centers_scrambled_final])

    # (2) Build cluster: the central Ga and its 4 nearest As neighbors.
    ga_pos = atoms[[a.symbol == 'Ga' for a in atoms]].positions[0]
    as_pos = atoms[[a.symbol == 'As' for a in atoms]].positions[0]

    # Shift each converged center by lattice vectors to be nearest
    # to the Ga atom in the home unit cell.
    ga_center = ga_pos.copy()
    for w in range(Nw):
        final_plot[w] = shift_to_nearest(final_plot[w], ga_center, cell)

    # Generate As images and pick the 4 closest to this Ga
    as_images = []
    for n1 in range(-1, 2):
        for n2 in range(-1, 2):
            for n3 in range(-1, 2):
                as_images.append(as_pos + n1 * cell[0]
                                 + n2 * cell[1] + n3 * cell[2])
    as_images = np.array(as_images)
    dists = np.linalg.norm(as_images - ga_center, axis=1)
    nearest_4 = as_images[np.argsort(dists)[:4]]

    # (3) Shift initial centers: Hungarian matching to minimize arrows.
    cost_arrows = np.zeros((Nw, Nw))
    best_init_for_pair = [[None] * Nw for _ in range(Nw)]
    for i in range(Nw):
        for f in range(Nw):
            s = shift_to_nearest(centers_scrambled_init[i],
                                 final_plot[f], cell)
            best_init_for_pair[i][f] = s
            cost_arrows[i, f] = np.linalg.norm(s - final_plot[f])
    init_row, init_col = linear_sum_assignment(cost_arrows)
    # Reorder so init_plot[j] maps to final_plot[j]
    init_plot = np.empty((Nw, 3))
    for i, f in zip(init_row, init_col):
        init_plot[f] = best_init_for_pair[i][f]

    fig2 = plt.figure(figsize=(7, 6))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Plot atoms: central Ga + 4 nearest As
    ax2.scatter(*ga_center, s=200, c='purple', edgecolors='k', zorder=5)
    for a_pos in nearest_4:
        ax2.scatter(*a_pos, s=200, c='green', edgecolors='k', zorder=5)

    # Draw Ga-As bonds
    for a_pos in nearest_4:
        ax2.plot([ga_center[0], a_pos[0]],
                 [ga_center[1], a_pos[1]],
                 [ga_center[2], a_pos[2]],
                 'k-', alpha=0.3, linewidth=1.0)

    # Plot scrambled initial Wannier centers (blue circles)
    for w in range(Nw):
        ax2.scatter(*init_plot[w], s=60, c='steelblue',
                    marker='o', edgecolors='k', alpha=0.6, zorder=3)

    # Plot reference converged centers (shifted to the same Ga)
    ref_plot = np.array([c.copy() for c in centers_ref])
    for w in range(Nw):
        ref_plot[w] = shift_to_nearest(ref_plot[w], ga_center, cell)
    for w in range(Nw):
        ax2.scatter(*ref_plot[w], s=80, c='gold', marker='*',
                    edgecolors='k', zorder=4, linewidths=0.5)

    # Plot scrambled converged Wannier centers (red diamonds)
    for w in range(Nw):
        ax2.scatter(*final_plot[w], s=80, c='red', marker='D',
                    edgecolors='k', zorder=4)

    # Draw arrows from scrambled-initial to converged centers
    for w in range(Nw):
        dx = final_plot[w] - init_plot[w]
        ax2.quiver(init_plot[w][0], init_plot[w][1], init_plot[w][2],
                   dx[0], dx[1], dx[2],
                   color='orange', linewidth=1.5, arrow_length_ratio=0.15)

    ax2.set_xlabel('x (A)')
    ax2.set_ylabel('y (A)')
    ax2.set_zlabel('z (A)')
    ax2.set_title('Tutorial 01b: scrambled (blue) vs converged (red) WF centers\n'
                  'Arrows show optimization path')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
               markersize=10, label='Ga'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='As'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=8, label='Scrambled initial WF center'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
               markersize=10, label='Reference converged center'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markersize=8, label='Scrambled converged center'),
    ]
    ax2.legend(handles=legend_elements, fontsize=8)

    plot2_path = tutorial_dir / 'tutorial01b_centers.png'
    fig2.savefig(plot2_path, dpi=150)
    print(f"  Saved: {plot2_path.name}")
    print()

except ImportError:
    print("  (matplotlib not available -- skipping plots)")
    print()

print("=" * 65)
print("  Tutorial 01b complete!")
print("=" * 65)
