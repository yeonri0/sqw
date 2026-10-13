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
L = 8
J = 1.0
H = np.zeros((2**L, 2**L), dtype=complex)
for i in range(L - 1):
    H += J * (op(Sx, i, L) @ op(Sx, i + 1, L)+ op(Sy, i, L) @ op(Sy, i + 1, L)+ op(Sz, i, L) @ op(Sz, i + 1, L))

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
plt.savefig("sq_ed.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 5))
plt.pcolormesh(q_values, omega,S_qw.T, cmap="turbo",shading="auto")
plt.xlabel(r"$q$")
plt.ylabel(r"$\omega/J$")
plt.colorbar(label=r"$S^{zz}(q,\omega)$")
plt.xticks(
    [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
)
plt.tight_layout()
plt.ylim(0, 4)
plt.savefig("sqw_ed_L8.png", dpi=300, bbox_inches="tight")
plt.show()