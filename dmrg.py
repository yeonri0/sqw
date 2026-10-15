"""
DMRG収束確認

"""

import numpy as np
import quimb as qu
import inspect
import quimb.tensor as qtn
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.labelsize": 16,    # x軸・y軸のラベル
    "xtick.labelsize": 14,   # x軸の目盛り
    "ytick.labelsize": 14,   # y軸の目盛り
    "axes.titlesize": 14,    # グラフタイトル
    "legend.fontsize": 12,   # 凡例
    "figure.titlesize": 14,  # Figure全体のタイトル
})

# Heisenberg Hamiltonian
L = 10
J = 1.0
H = qtn.MPO_ham_heis(L,j=J,cyclic=False)    # HeisenbergハミルトニアンをMPO形式で作る。周期境界条件にせず、両端を結ばない。OBC（開境界条件）

cutoff = 1e-10
tol_energy = 1e-8
max_sweeps = 8


# DMRGを実行し、スイープごとのエネルギーを記録する関数
def run_dmrg(D, p0=None):
    dmrg = qtn.DMRG2(
        H,
        bond_dims=[D],
        cutoffs=cutoff,
        p0=p0,
    )
    energy_history = []

    converged_count = 0

    for i in range(max_sweeps):
        dmrg.solve(max_sweeps=1, tol=0, verbosity=0)
        energy_history.append(float(dmrg.energy))

        if i > 0:
            delta_E = abs(energy_history[i] - energy_history[i - 1])
            print(
                f"Sweep {i + 1}: "
                f"E = {energy_history[i]:.12f}, "
                f"|delta E| = {delta_E:.3e}"
            )

            if delta_E < tol_energy:
                converged_count += 1
            else:
                converged_count = 0

            if converged_count >= 5:
                print("Energy convergence reached.")
                break
    return dmrg, energy_history


# ==================================================
# 1. スイープごとのエネルギー変化
# ==================================================

dmrg, energy_history = run_dmrg(D=40)


# ==================================================
# 2. 初期状態を変えて比較
# ==================================================

initial_states = [
    qtn.MPS_rand_state(
        L, bond_dim=4, phys_dim=2,
        normalize=True, seed=seed
    )
    for seed in [1, 2, 3]
]

initial_energies = []

for i in range(len(initial_states)):
    dmrg, history = run_dmrg(D=40, p0=initial_states[i])
    initial_energies.append(dmrg.energy)

print("\nInitial-state comparison:")
for i in range(len(initial_energies)):
    print(f"Initial state {i + 1}: E = {initial_energies[i]:.12f}")


# ==================================================
# 3. ボンド次元を変えて比較
# ==================================================

bond_dims = [4, 8, 12, 16, 20]
bond_energies = []

for i in range(len(bond_dims)):
    dmrg, history = run_dmrg(D=bond_dims[i])
    bond_energies.append(dmrg.energy)

print("\nBond-dimension comparison:")
for i in range(len(bond_dims)):
    print(
        f"D = {bond_dims[i]:2d}: "
        f"E = {bond_energies[i]:.12f}"
    )

print("\nEnergy differences:")
for i in range(1, len(bond_dims)):
    delta_E = abs(bond_energies[i] - bond_energies[i - 1])
    print(
        f"D = {bond_dims[i - 1]} -> {bond_dims[i]}: "
        f"|delta E| = {delta_E:.3e}"
    )


import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. スイープごとのエネルギー変化
# ==========================================

dmrg, energy_history = run_dmrg(D=40)

sweeps = np.arange(1, len(energy_history) + 1)
delta_E = np.abs(np.diff(energy_history))

plt.figure()
plt.plot(sweeps[1:], delta_E, marker="o")
plt.yscale("log")
plt.xlabel("Sweep number n")
plt.ylabel(r"$|E_n - E_{n-1}|$")
plt.title("Energy convergence D = 40")
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence_D40.png", dpi=300, bbox_inches="tight")
plt.show()


# ==========================================
# 2. 初期状態ごとのエネルギー収束
# ==========================================

initial_states = [
    qtn.MPS_rand_state(
        L, bond_dim=4, phys_dim=2,
        normalize=True, seed=seed
    )
    for seed in [1, 2, 3]
]

initial_histories = []

for i in range(len(initial_states)):
    dmrg, history = run_dmrg(D=40, p0=initial_states[i])
    initial_histories.append(history)

plt.figure()

for i in range(len(initial_histories)):
    history = initial_histories[i]
    sweeps = np.arange(1, len(history) + 1)
    plt.plot(sweeps, history, marker="o", label=f"Initial state {i + 1}")

plt.xlabel("Sweep number n")
plt.ylabel("Ground-state energy")
plt.title("Convergence from different initial states")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence_initial_states.png", dpi=300, bbox_inches="tight")
plt.show()


# ==========================================
# 3. ボンド次元ごとの最終エネルギー
# ==========================================

bond_dims = [4, 8, 12, 16, 20]
bond_energies = []

for i in range(len(bond_dims)):
    dmrg, history = run_dmrg(D=bond_dims[i])
    bond_energies.append(history[-1])

plt.figure()
plt.plot(bond_dims, bond_energies, marker="o")
plt.xlabel("Bond dimension D")
plt.ylabel("Ground-state energy")
plt.title("Energy vs. bond dimension")
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_vs_bond_dimension.png", dpi=300, bbox_inches="tight")
plt.show()
