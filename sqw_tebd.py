"""
S(q,w) in 1次元 Heisenberg, L = 64 site using TEBD
中性子非弾性散乱は、磁性体のスピンの励起を、波数𝑞とエネルギー𝜔ごとに分けて測る。
測られる量を動的構造因子（dynamical structure factor, DSF）S(q,w)と呼ぶ。
S(q,w)は、位置が𝑟、時刻が𝑡だけ離れた 2 つのスピンの相関関数G(r,t)の Fourier変換である。

TEBD
│
├─ H
├─ ground state |ψ₀⟩ , E0     ← DMRG
│
├─ |φ(0)⟩ = Sᶻc|ψ₀⟩
│
├─ |φ(t)⟩ = exp(-iHt)|φ(0)⟩  ← TEBD
│
├─ G(r,t)
│
├─ Fourier transform -> G(q,t)
│
└─ Fourier transform -> S(q,ω)

鎖の長さは、乱れが端に届く時刻から決める。
Fourier 変換では時間と距離に窓関数を掛け、和を任意のq, wで評価する。
分解能は、𝜎𝜔 = 3/tmaxである。

"""

import numpy as np
import quimb as qu
import quimb.tensor as qtn

# Heisenberg Hamiltonian
L = 64
J = 1.0
H = qtn.MPO_ham_heis(L,j=J,cyclic=False)    # HeisenbergハミルトニアンをMPO形式で作る。周期境界条件にせず、両端を結ばない。OBC（開境界条件）

c = L // 2      # サイト中央
t_max = 20
dt = 0.05


# 2サイトDMRG
dmrg = qtn.DMRG2(H,bond_dims=[40, 60, 80, 100, 120],cutoffs=1e-10)  # DMRG計算を行うためのオブジェクト   # ボンド次元の上限を順番に変える。異なるボンド次元でエネルギーが変わらないかで収束を判定
dmrg.solve(tol=1e-8,verbosity=1)    # tolは収束判定の許容誤差を指定

# Ground state
psi0 = dmrg.state
E0 = dmrg.energy
print("Ground-state energy =", E0)
print("Energy per site      =", E0 / L)


# TEBDでG(n,c,t)を求める     15ページ参照
Sz = qu.spin_operator("z")
psi_exc = psi0.copy()
phi_0 = psi_exc.gate_(Sz, where=c, contract=True)       # |φ(0)⟩ = Sᶻc|ψ₀⟩

H_tebd = qtn.ham_1d_heis(L, j=J, cyclic=False)      # まずは1次元Heisenberg鎖のハミルトニアンを用意
tebd = qtn.TEBD(phi_0, H_tebd, dt=dt, split_opts={"max_bond": 128, "cutoff": 1e-10}, progbar=True)   # phi0 や H_tebd などを保持し、TEBD計算を行うためのオブジェクト
times = np.arange(0, t_max + dt, dt)                 # 等間隔の時刻の配列を生成
states = tebd.at_times(times, order=2)               # 2次のSuzuki-Trotter分解を使って時間発展を近似

# 時間ごとにG(n,t)を計算　ただし中心はc = L / 2
G = np.zeros((L, len(times)), dtype=complex)
for it in range(len(times)):
    t = times[it]
    phi_t = next(states)                                        # |φ(t)⟩ = U(t)|φ(0)⟩
    for n in range(L):
        phi_tmp = phi_t.copy()
        phi_tmp.gate_(Sz, where=n)                              # Sᶻc|φ(t)⟩
        G[n, it] = np.exp(1j * E0 * t) * (psi0.H @ phi_tmp)     # G(n,c,t) = e^iE0t <psi0| Sᶻc|φ(t)⟩

# DMRG・TEBDで計算した結果を保存
np.savez("correlation.npz", G=G, times=times, E0=E0)


import matplotlib.pyplot as plt

# 保存した結果を読み込む、ここからフーリエ変換する
data = np.load("correlation.npz")
G = data["G"]
times = data["times"]
E0 = float(data["E0"])

# G(r,t) のパワープロット（光円錐）
r = np.arange(L) - c
plt.figure(figsize=(8, 6))
limit = np.max(np.abs(np.real(G)))
from matplotlib.colors import TwoSlopeNorm
plt.pcolormesh(
    r,
    times,
    np.real(G).T,
    shading="auto",
    cmap="RdBu",
    norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
)
plt.xlabel(r"$r=n-c$")
plt.ylabel(r"$tJ$")
plt.colorbar(label=r"$G(r,t)$")
plt.xlim(-L // 2, L // 2)
plt.tight_layout()
plt.savefig("correlation_lightcone.png", dpi=300, bbox_inches="tight")
plt.show()



# Spatial Fourier transformでG(q,t)を作る
r = np.arange(L) - c
q_vals = np.linspace(0, 2 * np.pi, 2 * L + 1)
Gqt = np.zeros((len(q_vals), len(times)), dtype=complex)
for iq in range(len(q_vals)):
    q = q_vals[iq]

    Gqt[iq, :] = np.sum(np.exp(-1j * q * r[:, None]) * G,axis=0)

# Time Eourier transform
dt_obs = dt 

# Gaussian window
sigma_t = t_max / 3
window = np.exp(-times**2 / (2 * sigma_t**2))
Gqt_windowed = Gqt * window[None, :]

omega_vals = np.linspace(0, 4, 400)
Sqw = np.zeros((len(q_vals), len(omega_vals)))

for iq in range(len(q_vals)):
    for iw in range(len(omega_vals)):
        omega = omega_vals[iw]

        Sqw[iq, iw] = dt_obs / (2 * np.pi) * (Gqt_windowed[iq, 0] + 2 * np.real(np.sum(Gqt_windowed[iq, 1:]* np.exp(1j * omega * times[1:]))))


iq = np.argmin(np.abs(q_vals - np.pi / 2))
plt.figure(figsize=(8, 5))
plt.plot(omega_vals, Sqw[iq, :])
plt.xlabel(r"$\omega/J$")
plt.ylabel(r"$S(\pi/2,\omega)$")
plt.tight_layout()
plt.savefig("sw_tebd.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(8, 5))
plt.pcolormesh(q_vals, omega_vals, Sqw.T, cmap="viridis", shading="auto")
plt.xlabel(r"$q$")
plt.ylabel(r"$\omega/J$")
plt.colorbar(label=r"$S^{zz}(q,\omega)$")
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi], [r"$0$", r"${\pi}/{2}$", r"$\pi$"])
plt.ylim(0, 4)
plt.tight_layout()
plt.savefig("sqw_tebd_L64.png", dpi=300, bbox_inches="tight")
plt.show()