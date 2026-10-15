"""
S(q,w) in 1次元 Heisenberg, L = 8 site using ED 
中性子非弾性散乱は、磁性体のスピンの励起を、波数𝑞とエネルギー𝜔ごとに分けて測る。
測られる量を動的構造因子（dynamical structure factor, DSF）S(q,w)と呼ぶ。
S(q,w)は、位置が𝑟、時刻が𝑡だけ離れた 2 つのスピンの相関関数G(r,t)の Fourier変換である。

ED
│
├─ H
├─ ground state |ψ₀⟩
│
└─ S(q,ω)

"""
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

plt.rcParams.update({
    "axes.labelsize": 16,    # x軸・y軸のラベル
    "xtick.labelsize": 14,   # x軸の目盛り
    "ytick.labelsize": 14,   # y軸の目盛り
    "axes.titlesize": 14,    # グラフタイトル
    "legend.fontsize": 12,   # 凡例
    "figure.titlesize": 14,  # Figure全体のタイトル
})

up = np.array([1, 0], dtype=complex)  # |↑⟩
dn = np.array([0, 1], dtype=complex)  # |↓⟩

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]], dtype=complex)
Sx, Sy, Sz = sx / 2, sy / 2, sz / 2

def op(A, i, N):
    # サイト i（1始まり）に A を置き、残りのサイトを I で埋める
    # 例えば、op(Sx, 2, 3) = I ⊗ Sx ⊗ I
    return reduce(np.kron, [A if j == i else I2 for j in range(1, N + 1)])


# Heisenberg Hamiltonian
L = 10
J = 1.0
H = np.zeros((2**L, 2**L), dtype=complex)

for i in range(1, L):
    H += J * (
        op(Sx, i, L) @ op(Sx, i + 1, L)
        + op(Sy, i, L) @ op(Sy, i + 1, L)
        + op(Sz, i, L) @ op(Sz, i + 1, L)
    )

# Hamiltonianを対角化
E, V = np.linalg.eigh(H)

# 基底状態
E0 = E[0]
psi0 = V[:, 0]

print("Ground-state energy =", E0)

# S_q^z, S(q)=S_q, S(q,w)=S_qwを作る 11ページ参考
q_values = np.linspace(0, 2*np.pi, 101)
omega = np.linspace(0, 4 * J, 500)

eta = 0.05

S_q = np.zeros(len(q_values))
S_qw = np.zeros((len(q_values), len(omega)))

# 時間発展を追わず、各エネルギー固有状態への重なりからスペクトルを直接求める。ただしデルタ関数を Lorentzian で近似。
for iq, q in enumerate(q_values):

    # S_q^z
    Sq = sum(np.exp(1j * q * r) * op(Sz, r, L) for r in range(L)) / np.sqrt(L)

    # S_q^z |0>
    phi = Sq @ psi0

    # <m|S_q^z|0>
    amplitude = V.conj().T @ phi

    # |<m|S_q^z|0>|^2
    weight = np.abs(amplitude)**2

    # S(q)
    S_q[iq] = np.sum(weight)

    # S(q, omega)
    omega_m = E - E0

    S_qw[iq] = sum(weight[m] * (eta / np.pi/ ((omega - omega_m[m])**2 + eta**2)) for m in range(len(E))) # Lorentzian近似


plt.figure(figsize=(7, 5))
plt.plot(q_values, S_q)
plt.xlabel(r"$q$")
plt.ylabel(r"$S(q)$")

plt.xticks(
    [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
)
plt.tight_layout()
plt.savefig("sq_ed_L10.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
q_exact = np.linspace(0, 2 * np.pi, 500)
omega_exact1 = np.pi * J * np.abs(np.sin(q_exact / 2))
omega_exact2 = np.pi * J / 2 * np.abs(np.sin(q_exact))
plt.plot(q_exact,omega_exact1, "b-", linewidth=1.8, label=r"$\omega_{\mathrm{U}}(q)=\pi J|sin\frac{q}{2}|$")
plt.plot(q_exact,omega_exact2, "y-", linewidth=1.8, label=r"$\omega_{\mathrm{L}}(q)=\frac{\pi J}{2}|\sin q|$")
plt.legend()
c =plt.pcolormesh(q_values, omega, S_qw.T/np.max(S_qw), cmap="turbo",shading="auto")
plt.xlabel(r"$q$")
plt.ylabel(r"$\omega/J$")
cbar = plt.colorbar(c, label=r"$S^{zz}(q,\omega)$ normalized")
cbar.set_label(r"$S^{zz}(q,\omega)$ normalized", fontsize=16)
c.set_clim(0, 1)
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.ax.tick_params(labelsize=12)
plt.xlim(0, 2*np.pi)
plt.xticks(
    [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
)
plt.ylim(0.0, 3.5)
plt.tight_layout()
plt.title("Dynamical structure factor L=10 ED")
plt.savefig("sqw_ed_L10.png", dpi=300, bbox_inches="tight")
plt.show()