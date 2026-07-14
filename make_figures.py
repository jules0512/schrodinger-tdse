"""Regenerate every figure and the animation in figures/.

    python make_figures.py

Kept deliberately readable: it is also an example of how to *use* schrodinger.py.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import schrodinger as sch

np.seterr(all="ignore")

# ---- a calm, consistent house style --------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888", "figure.facecolor": "white",
})
DENS, RE, IM, WALL = "#2a9d8f", "#e76f51", "#4a6fa5", "#e9c46a"
FIG = "figures"

# ---- shared simulation set-up --------------------------------------------- #
L, k0, J = 1.0, 100.0, 256
x = np.linspace(0, L, J)
dx, dt, N = x[1] - x[0], 1e-6, 3000
V0 = k0**2                       # barrier height equal to the packet energy (E = V0)


def packet(sigma=0.05):
    return sch.gaussian_packet(x, 0.25 * L, sigma * L, k0)


def draw_state(ax, psi, V, title):
    """One tidy snapshot: filled density, thin Re/Im, shaded potential."""
    dens = np.abs(psi) ** 2
    dens = dens / dens.max() if dens.max() else dens
    s = max(np.abs(psi.real).max(), np.abs(psi.imag).max(), 1e-30)
    if V.max() > 0:
        ax.fill_between(x, 0, V / V.max(), color=WALL, alpha=0.5, lw=0, label="V(x)")
    ax.plot(x, psi.real / s, color=RE, lw=0.8, alpha=0.7, label="Re ψ")
    ax.plot(x, psi.imag / s, color=IM, lw=0.8, alpha=0.7, label="Im ψ")
    ax.fill_between(x, 0, dens, color=DENS, alpha=0.25, lw=0)
    ax.plot(x, dens, color=DENS, lw=2, label="|ψ|²")
    ax.set(xlabel="x", title=title, ylim=(-1.15, 1.15), xlim=(0, L))


def snapshots(psi, V, title, fname):
    picks = [0, len(psi) // 3, 2 * len(psi) // 3]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True, constrained_layout=True)
    for ax, n in zip(axes, picks):
        draw_state(ax, psi[n], V, f"t = {n * dt:.1e}")
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.suptitle(title, fontsize=13)
    fig.savefig(f"{FIG}/{fname}")
    plt.close(fig)
    print(f"  {fname}")


def spacetime(psi, title, fname, barrier_x=None):
    """|ψ(x,t)|² as a heat map: read time upward, space rightward."""
    dens = np.abs(psi) ** 2
    rows = np.linspace(0, len(psi) - 1, 400).astype(int)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    im = ax.imshow(dens[rows], origin="lower", aspect="auto", cmap="magma",
                   extent=[0, L, 0, len(psi) * dt])
    if barrier_x is not None:
        ax.axvline(barrier_x, color="white", lw=1, ls="--", alpha=0.7)
    ax.set(xlabel="position  x", ylabel="time  t", title=title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="|ψ(x, t)|²", shrink=0.85)
    fig.savefig(f"{FIG}/{fname}")
    plt.close(fig)
    print(f"  {fname}")


# --------------------------------------------------------------------------- #
def main():
    print(f"setup: J={J}, dx={dx:.2e}, dt={dt:.1e}, N={N}")

    # 1+2. Free particle: snapshots + space-time map
    free = sch.evolve_crank_nicolson(packet(), sch.free(x), dx, dt, N)
    snapshots(free, sch.free(x), "Free particle — a spreading wave packet", "free_snapshots.png")
    spacetime(free, "Free particle: the packet glides and spreads", "spacetime_free.png")

    # 3. Norm: explicit drifts, Crank-Nicolson does not (fine-dt experiment)
    dt2, N2 = dx**2 / 50, 2500
    cn = sch.evolve_crank_nicolson(packet(), sch.free(x), dx, dt2, N2)
    ex = sch.evolve_explicit(packet(), sch.free(x), dx, dt2, N2)
    t = np.arange(N2 + 1) * dt2
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.semilogy(t, np.abs(sch.norm(ex, dx) - 1), color=RE, label="explicit scheme")
    ax.semilogy(t, np.abs(sch.norm(cn, dx) - 1), color=DENS, label="Crank-Nicolson")
    ax.set(xlabel="time t", ylabel="|total probability − 1|",
           title="Probability conservation: Crank-Nicolson versus the explicit scheme")
    ax.legend()
    fig.savefig(f"{FIG}/norm_comparison.png")
    plt.close(fig)
    print(f"  norm_comparison.png  (CN {np.abs(sch.norm(cn,dx)-1).max():.0e}, "
          f"EX {np.abs(sch.norm(ex,dx)-1).max():.0e})")

    # 4+5. Barrier: snapshots, space-time map, R/T
    Vb = sch.barrier(x, V0, 0.03, 0.5)
    bar = sch.evolve_crank_nicolson(packet(), Vb, dx, dt, N)
    snapshots(bar, Vb, "Tunnelling through a barrier (E = V₀)", "barrier_snapshots.png")
    spacetime(bar, "Tunnelling: reflected (←) and transmitted (→) parts", "spacetime_barrier.png", 0.5)

    R, T = sch.reflection_transmission(bar, x, 0.5)
    tt = np.arange(N + 1) * dt
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(tt, R, color=RE, label="R (reflected)")
    ax.plot(tt, T, color=DENS, label="T (transmitted)")
    ax.plot(tt, R + T, "--", color="#444", lw=1, label="R + T")
    ax.set(xlabel="time t", ylabel="probability", ylim=(-0.05, 1.1),
           title="Reflection, transmission, and conservation of probability")
    ax.legend()
    fig.savefig(f"{FIG}/rt_time.png")
    plt.close(fig)
    print("  rt_time.png")

    # 6. Exponential decay of transmission with barrier width (E/V0 = 0.5)
    Vh = 2 * V0
    kappa = np.sqrt(Vh - k0**2)         # = sqrt(V0 - E)
    widths = np.linspace(0.01, 0.05, 8)
    Ts = []
    for w in widths:
        Vw = sch.barrier(x, Vh, w, 0.5)
        p = sch.evolve_crank_nicolson(packet(sigma=0.09), Vw, dx, dt, N)
        Ts.append(sch.reflection_transmission(p[-1], x, 0.5 + w / 2)[1])
    Ts = np.array(Ts)
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.semilogy(widths, Ts, "o", color=IM, ms=7, label="simulation")
    ref = Ts[0] * np.exp(-2 * kappa * (widths - widths[0]))
    ax.semilogy(widths, ref, "--", color="#444", lw=1.2,
                label=rf"$\propto e^{{-2\kappa l}},\ \kappa=\sqrt{{V_0-E}}={kappa:.0f}$")
    ax.set(xlabel="barrier width  l", ylabel="transmission  T",
           title="Under-barrier tunnelling (E/V₀ = 0.5): exponential decay")
    ax.legend()
    fig.savefig(f"{FIG}/tunnel_decay.png")
    plt.close(fig)
    slope = np.polyfit(widths, np.log(Ts), 1)[0]
    print(f"  tunnel_decay.png  (slope {slope:.0f} vs theory {-2*kappa:.0f})")

    # 7. Hero animation
    frames = np.linspace(0, N, 150).astype(int)
    Vn = Vb / Vb.max()
    fig, ax = plt.subplots(figsize=(7.2, 4), constrained_layout=True)
    ax.fill_between(x, 0, Vn, color=WALL, alpha=0.5, lw=0)
    dline, = ax.plot([], [], color=DENS, lw=2.2, label="|ψ|²")
    rline, = ax.plot([], [], color=RE, lw=0.7, alpha=0.6, label="Re ψ")
    iline, = ax.plot([], [], color=IM, lw=0.7, alpha=0.6, label="Im ψ")
    ax.set(xlim=(0, L), ylim=(-1.15, 1.15), xlabel="x",
           title="Quantum tunnelling: a wave packet meets a barrier")
    ax.legend(loc="upper right", fontsize=8)

    def update(k):
        psi = bar[frames[k]]
        d = np.abs(psi) ** 2
        d = d / d.max() if d.max() else d
        s = max(np.abs(psi.real).max(), np.abs(psi.imag).max(), 1e-30)
        dline.set_data(x, d)
        rline.set_data(x, psi.real / s)
        iline.set_data(x, psi.imag / s)
        return dline, rline, iline

    FuncAnimation(fig, update, frames=len(frames), blit=True).save(
        f"{FIG}/tunnel.gif", writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"  tunnel.gif  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
