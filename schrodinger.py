"""
schrodinger.py
==============

Building blocks for solving the one-dimensional time-dependent Schrödinger
equation

    i ∂ψ/∂t = -∂²ψ/∂x² + V(x) ψ          (ħ = 1, m = 1/2)

for a wave packet confined to a box with hard walls. The wave function is
represented as a NumPy array sampled on a uniform grid, a potential is an array
of the same shape, and the two integrators advance that array in time. Each
function is kept deliberately short so that it can be read against the equation
it implements.

Example
-------
    import numpy as np
    import schrodinger as sch

    x = np.linspace(0, 1, 256)
    dx = x[1] - x[0]

    psi0 = sch.gaussian_packet(x, x0=0.25, sigma=0.05, k0=100)
    V = sch.barrier(x, height=100**2, width=0.03, center=0.5)

    psi = sch.evolve_crank_nicolson(psi0, V, dx, dt=1e-6, n_steps=3000)
    # psi[n] is the wave function at time n * dt
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# --------------------------------------------------------------------------- #
# Initial state
# --------------------------------------------------------------------------- #
def gaussian_packet(x, x0, sigma, k0):
    """Return a normalised Gaussian wave packet on the grid x.

        psi(x) = exp(i k0 x) * exp(-(x - x0)^2 / (2 sigma^2))

    The factor exp(i k0 x) gives the packet a mean momentum k0, hence a group
    velocity 2 k0 (since m = 1/2). The packet is normalised so that the total
    probability sum(|psi|^2) dx equals one, and it is set to zero at the walls.
    """
    dx = x[1] - x[0]
    envelope = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    psi = np.exp(1j * k0 * x) * envelope
    psi[0] = 0.0
    psi[-1] = 0.0
    total_probability = np.sum(np.abs(psi) ** 2) * dx
    return psi / np.sqrt(total_probability)


# --------------------------------------------------------------------------- #
# Potentials. Each function returns V(x) sampled on the grid.
# --------------------------------------------------------------------------- #
def free(x):
    """Return the zero potential of a free particle."""
    return np.zeros_like(x)


def step(x, height, position):
    """Return a potential step that rises to `height` for x > `position`."""
    V = np.zeros_like(x)
    V[x > position] = height
    return V


def barrier(x, height, width, center):
    """Return a rectangular barrier of given `height` and `width`."""
    V = np.zeros_like(x)
    V[np.abs(x - center) <= width / 2] = height
    return V


def double_well(x, height, width, gap):
    """Return two barriers separated by `gap`, forming a double well."""
    middle = (x[0] + x[-1]) / 2
    left_barrier = barrier(x, height, width, middle - gap / 2)
    right_barrier = barrier(x, height, width, middle + gap / 2)
    return np.maximum(left_barrier, right_barrier)


# --------------------------------------------------------------------------- #
# Time integrators
# --------------------------------------------------------------------------- #
def evolve_explicit(psi0, V, dx, dt, n_steps):
    """Advance the wave function with the explicit forward-time scheme.

    A forward difference in time, with the spatial derivative evaluated at the
    current step, gives

        psi_j^{n+1} = psi_j^n
                      + i (dt / dx^2) (psi_{j+1}^n - 2 psi_j^n + psi_{j-1}^n)
                      - i dt V_j psi_j^n .

    The scheme is a single explicit update and requires no linear algebra, but
    for the Schrödinger equation it is unstable: the total probability grows
    rather than staying constant. It is included so that this behaviour can be
    observed directly and compared with Crank-Nicolson.

    Returns an array of shape (n_steps + 1, len(x)), where row n holds the wave
    function at time n * dt.
    """
    n_points = len(psi0)
    psi = np.zeros((n_steps + 1, n_points), dtype=complex)
    psi[0] = psi0

    for n in range(n_steps):
        current = psi[n]
        laplacian = current[2:] - 2 * current[1:-1] + current[:-2]
        psi[n + 1, 1:-1] = (
            current[1:-1]
            + 1j * dt / dx ** 2 * laplacian
            - 1j * dt * V[1:-1] * current[1:-1]
        )
        # The endpoints are left at zero, which enforces the hard walls.

    return psi


def evolve_crank_nicolson(psi0, V, dx, dt, n_steps):
    """Advance the wave function with the Crank-Nicolson scheme.

    The Hamiltonian is applied half at the old time and half at the new one,

        (1 + i dt/2 H) psi^{n+1} = (1 - i dt/2 H) psi^n ,

    with the discrete Hamiltonian

        (H psi)_j = -(psi_{j+1} - 2 psi_j + psi_{j-1}) / dx^2 + V_j psi_j .

    Introducing r = i dt / (2 dx^2), this is a tridiagonal linear system

        A psi^{n+1} = B psi^n

    with constant matrices A and B, since the potential does not depend on time.
    The interior points are the unknowns, and the walls remain at zero. The
    scheme is stable for any time step and conserves the total probability.

    Returns an array of shape (n_steps + 1, len(x)), where row n holds the wave
    function at time n * dt.
    """
    n_points = len(psi0)
    r = 1j * dt / (2 * dx ** 2)

    # Diagonals of the two operators, evaluated over the whole grid.
    diagonal_A = (1 + 2 * r) + 1j * dt / 2 * V
    diagonal_B = (1 - 2 * r) - 1j * dt / 2 * V

    # Assemble A for the interior unknowns as an explicit tridiagonal matrix.
    interior = slice(1, n_points - 1)
    n_interior = n_points - 2
    off_diagonal = np.full(n_interior - 1, -r)
    A = diags(
        [off_diagonal, diagonal_A[interior], off_diagonal],
        offsets=[-1, 0, 1],
        format="csc",
    )

    psi = np.zeros((n_steps + 1, n_points), dtype=complex)
    psi[0] = psi0

    for n in range(n_steps):
        right_hand_side = diagonal_B * psi[n]
        right_hand_side[interior] += r * (psi[n, 2:] + psi[n, :-2])
        psi[n + 1, interior] = spsolve(A, right_hand_side[interior])

    return psi


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def norm(psi, dx):
    """Return the total probability sum(|psi|^2) dx.

    Works on a single state or on a full history of shape (n_steps + 1, len(x)),
    in which case an array of values, one per time step, is returned.
    """
    return np.sum(np.abs(psi) ** 2, axis=-1) * dx


def reflection_transmission(psi, x, split):
    """Return the reflected and transmitted probabilities about x = `split`.

    The reflected probability R is the integral of |psi|^2 to the left of
    `split`, and the transmitted probability T the integral to the right, each
    divided by the total probability so that R + T = 1. A single state or a
    full history may be passed.
    """
    dx = x[1] - x[0]
    density = np.abs(psi) ** 2
    split_index = int(np.argmin(np.abs(x - split)))
    total = density.sum(axis=-1) * dx
    reflected = density[..., :split_index].sum(axis=-1) * dx / total
    transmitted = density[..., split_index:].sum(axis=-1) * dx / total
    return reflected, transmitted
