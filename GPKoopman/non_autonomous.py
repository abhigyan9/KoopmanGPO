# Methods for Non-Autonomous System definition and simulation
import torch

# Duffing Oscillator with Control


def fc_DO(x, u, params=None):
    # params.device = x.device
    if params is None:
        params = torch.tensor([-0.2, 4., -1., 1.])

    dx0 = x[1]
    dx1 = params[0] * x[1] + params[1] * x[0] + params[2] * x[0] ** 3 + u
    return torch.tensor([dx0, dx1], dtype=torch.float64, device=x.device)


# Nonlinear Simpled Damped Pendulum with Control
def fc_SDP(x, u, params=None):
    # Common param names = length, damping coefficient
    if params is None:
        params = torch.tensor([1., 0.2])

    g = 9.81
    dx0 = x[1]
    dx1 = -(g/params[0]) * torch.sin(x[0]) - params[1] * x[1] + u
    return torch.tensor([dx0, dx1], dtype=torch.float64)


# Inverted Pendulum on a cart with horizontal force on cart
def fc_PoC(x, u, params=None):
    if params is None:
        params = torch.tensor([0.4, 1., 9.81, 0.5, 6., 0.1/12])

    m, M, g, l, b, I = params
    sigma = -(l * m * torch.cos(x[0])) ** 2 + \
        (m * l) ** 2 + (m * M * l ** 2) + I * (m + M)
    dx0 = x[1]
    dx1 = m * l * ((u - b * x[3]) * torch.cos(x[0]) + (m + M) * g * torch.sin(
        x[0]) - 0.5 * m * l * (x[1] ** 2) * torch.sin(2 * x[0])) / sigma
    dx2 = x[3]
    dx3 = (u * (I + m * l ** 2) - I * b * x[3] - l * ((m * l) ** 2 + I * m) * torch.sin(
        x[0]) * (x[1] ** 2) - b * m * x[3] * (l ** 2) + 0.5 * g * torch.sin(2 * x[0]) * ((m * l) ** 2)) / sigma
    return torch.tensor([dx0, dx1, dx2, dx3], dtype=torch.float64, device=x.device)


def sim_RK4_nonautonomous(fx, x0, ts, num_steps, u, params=None):
    """
    Runge-Kutta 4th Order Simulation for Non-Autonomous Dynamic Systems

    Arguments:
        fx: Callable function of the form fx(x, u, params) defining the dynamics.
        x0: Initial state as a PyTorch tensor of shape (n,).
        ts: Time step (float).
        num_steps: Number of simulation steps (int).
        u: Time-varying input, a tensor of shape (input_dim, num_steps).
        params: Additional parameters (default=torch.tensor([1.])).

    Returns:
        states: Simulated states of shape (n, num_steps).
    """
    n = x0.shape[0]
    states = torch.zeros((n, num_steps), dtype=torch.float64)
    states[:, 0] = x0

    for t in range(num_steps - 1):
        x = states[:, t]
        u_t = u[:, t]  # Input at time step t

        k1 = fx(x, u_t, params)
        k2 = fx(x + (ts/2)*k1, u_t, params)
        k3 = fx(x + (ts/2)*k2, u_t, params)
        k4 = fx(x + ts*k3, u_t, params)

        states[:, t+1] = x + (ts/6)*(k1 + 2*k2 + 2*k3 + k4)

    return states


def sim_LTI_nonautonomous(x0, A, B, C, u, num_steps, ts=0.01, disc='analytic', x0cv=None):
    """
    Forward stepping simulation of a non-autonomous Linear Time-Invariant system.

    Arguments:
        x0: Initial state as a PyTorch tensor of shape (n,).
        A: State transition matrix (n x n).
        B: Input matrix (n x p).
        C: Output matrix (m x n).
        u: Time-varying input, a tensor of shape (p, num_steps).
        num_steps: Number of simulation steps (int).
        ts: Time step (float). Set to None for discrete-time systems.
        disc: Discretization Option (str), Set to 'Euler' for non-invertible A.
        x0cv: Initial state covariance (optional, n x n tensor).

    Returns:
        x, y: States and outputs over time.
        Optionally: xcv, ycv if covariance tracking is needed.
    """
    m, n = C.shape
    p = B.shape[1]

    x = torch.zeros((n, num_steps))
    y = torch.zeros((m, num_steps))

    if x0.dim() == 1:
        x[:, 0] = x0
        y[:, 0] = C @ x0
    elif x0.dim() == 2:
        x[:, 0] = x0[:, 0]
        y[:, 0] = C @ x0[:, 0]
    else:
        raise ValueError('Expected x0 to be a 1D/2D tensor')

    if not ts is None:
        if disc == 'analytic':
            Ad = torch.linalg.matrix_exp(A * ts)
            Bd = torch.linalg.solve(
                A, (Ad - torch.eye(n, device=A.device)) @ B)
        else:
            Ad = A * ts + torch.eye(n, device=A.device)
            Bd = B * ts

    else:
        Ad = A
        Bd = B

    if x0cv is None:
        for k in range(num_steps-1):
            x[:, k+1] = Ad @ x[:, k] + Bd @ u[:, k]
            y[:, k] = C @ x[:, k]

        y[:, -1] = C @ x[:, -1]

        return x, y
    else:
        if x0cv.shape[0] is not n or x0cv.shape[1] is not n:
            raise ValueError('Expected x0cv to be a 2D tensor of shape (n,n)')

        xcv = torch.zeros((n, n, num_steps))
        ycv = torch.zeros((m, m, num_steps))
        xcv[:, :, 0] = x0cv

        for k in range(num_steps - 1):
            x[:, k+1] = Ad @ x[:, k] + Bd @ u[:, k]
            y[:, k] = C @ x[:, k]
            xcv[:, :, k+1] = Ad @ xcv[:, :, k] @ Ad.T
            ycv[:, :, k] = C @ xcv[:, :, k] @ C.T

        y[:, -1] = C @ x[:, -1]
        ycv[:, :, -1] = C @ xcv[:, :, -1] @ C.T

        return x, xcv, y, ycv
