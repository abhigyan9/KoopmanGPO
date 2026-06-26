# models and equations
import torch

def f_RR(x, params=None):
    # Common param names = alpha, beta, eps
    if params is None:
        params = torch.tensor([1.0, 1.0, 1e-3], dtype=torch.float64)
    alpha, beta, eps = params
    dx = -alpha * x[0] + beta / (eps + x[0] ** 2)
    return torch.tensor([dx], dtype=torch.float64)


def f_RBDP(x, params=None):
    # Common param names = kappa, omega, a
    if params is None:
        params = torch.tensor([0.1, 1.0, 0.5], dtype=torch.float64)
    kappa, omega, a = params
    dx0 = x[1]
    dx1 = -kappa * x[1] - (omega ** 2) * torch.sin(x[0]) + a / (1.0 + x[0] ** 2)
    return torch.tensor([dx0, dx1], dtype=torch.float64)


def f_IPP(x, params=None):
    # Common param names = r, K, a, h, n, eta, d
    if params is None:
        params = torch.tensor([1.0, 5.0, 1.0, 1.0, 2.0, 0.5, 0.3], dtype=torch.float64)
    r, K, a, h, n, eta, d = params
    prey, pred = x[0], x[1]
    denom = 1.0 + h * prey ** n
    interaction = (a * prey ** 2 / denom) * pred
    dx0 = r * prey * (1.0 - prey / K) - interaction
    dx1 = eta * interaction - d * pred
    return torch.tensor([dx0, dx1], dtype=torch.float64)


def f_UDO(x, params=None):
    # Common param names = alpha, beta, delta
    # Single attractor for alpha > 0
    # Clemson Duffing Oscillator = [1., -4., 0.]
    # General Oscillator setting is [1., 1., 1.]
    if params is None:
        params = torch.tensor([-4., 1., 0.])
    dx0 = x[1]
    dx1 = -params[0] * x[0] - params[1] * (x[0]**3) - params[2] * x[1]
    return torch.tensor([dx0, dx1], dtype=torch.float64)


def f_VDP(x, params=None):
    # Common param names = mu
    if params is None:
        params = torch.tensor([1.])
    dx0 = x[1]
    dx1 = params[0]*(1-x[0]**2)*x[1] - x[0]
    return torch.tensor([dx0, dx1], dtype=torch.float64)


def f_RVDP(x, params=None):
    # Common param names = mu
    if params is None:
        params = torch.tensor([1.])
    dx0 = -x[1]
    dx1 = x[0] - x[1] + ( (x[0] ** 2) * x[1] )
    return torch.tensor([dx0, dx1], dtype=torch.float64)


def f_SDP(x, params=None):
    # Common param names = length, damping coefficient
    if params is None:
        params = torch.tensor([1., 0.2])
    g = 9.81
    dx0 = x[1]
    dx1 = -(g/params[0]) * torch.sin(x[0]) - params[1] * x[1]
    return torch.tensor([dx0, dx1], dtype=torch.float64)


def f_Lorenz(x, params=None):
    # Common param names = sigma, beta, rho
    if params is None:
        params = torch.tensor([10., 8./3., 28.])
    dx0 = params[0] * (x[1] - x[0])
    dx1 = x[0] * (params[2] - x[2]) - x[1]
    dx2 = x[0] * x[1] - params[1] * x[2]
    return torch.tensor([dx0, dx1, dx2], dtype=torch.float64)


def f_Lorenz96(x, params=None):
    x = x.to(dtype=torch.float64)
    if params is None:
        params = x.new_tensor(5.0)
    
    dx = (
        (torch.roll(x, shifts=-1) - torch.roll(x, shifts=2))
        * torch.roll(x, shifts=1)
        - x
        + params[0]
    )

    return dx



def f_LotkaVolterra(x, params=None):
    # Common param names = alpha, beta, gamma, delta
    if params is None:
        params = torch.tensor([2./3., 4./3., 1., 1.])
    dx0 = params[0] * x[0] - params[1] * x[0] * x[1]
    dx1 = -params[2] * x[1] + params[3] * x[0] * x[1]
    return torch.tensor([dx0, dx1], dtype=torch.float64)


def f_PWL1(x, params=None):
    if params is None:
        params = torch.tensor([0.31, 0.94, -3., 0.32])
        # params = a, b, c, x*
    # assumed that sample-time if 1.
    if x[0] < params[3]:
        dx0 = (params[2] * x[0] - params[2] * params[3] - x[0]) / 1.
    else:
        dx0 = (params[1] * x[0] + params[0] -
               params[1] * params[3] - x[0]) / 1.

    return torch.tensor([dx0], dtype=torch.float64)


# Inverted Pendulum on a cart with horizontal force on cart
def f_uPoC(x, params=None):
    if params is None:
        params = torch.tensor([0.4, 1., 9.81, 0.5, 6., 0.1/12], dtype=torch.float64)

    m, M, g, l, b, I = params
    sigma = -(l * m * torch.cos(x[0])) ** 2 + \
        (m * l) ** 2 + (m * M * l ** 2) + I * (m + M)
    dx0 = x[1]
    dx1 = m * l * ((-b * x[3]) * torch.cos(x[0]) + (m + M) * g * torch.sin(
        x[0]) - 0.5 * m * l * (x[1] ** 2) * torch.sin(2 * x[0])) / sigma
    dx2 = x[3]
    dx3 = ( - I * b * x[3] - l * ((m * l) ** 2 + I * m) * torch.sin(
        x[0]) * (x[1] ** 2) - b * m * x[3] * (l ** 2) + 0.5 * g * torch.sin(2 * x[0]) * ((m * l) ** 2)) / sigma
    return torch.tensor([dx0, dx1, dx2, dx3], dtype=torch.float64, device=x.device)


def df_PWL(x, params=None):
    if params is None:
        params = torch.tensor([0.31, 0.94, -3., 0.32], dtype=torch.float64)
    if x[0] <= params[3]:
        xp0 = params[2] * (x[0] - params[3])
    else:
        xp0 = params[1] * (x[0] - params[3]) + params[0]
    return torch.tensor([xp0], dtype=torch.float64)


def df_scalarNL(x, params=None):
    x0p = -x[0] + (3 / ( 1 + (x[0] ** 2))) + (0.5 * torch.sin(2 * x[0]))
    return torch.tensor([x0p], dtype=torch.float64)


# def df_VDP_Surana2016(x, params=None):
#     if params is None:
#         params = torch.tensor([0.31, 0.94, -3., 0.32], dtype=torch.float64)
#     xp0 = x[0] - x[1] *


def sim_RK4(fx, x0, ts, num_steps, params=None):
    n = x0.shape[0]
    states = torch.zeros((n, num_steps), dtype=torch.float64)
    states[:, 0] = x0

    for t in range(num_steps - 1):
        x = states[:, t]

        k1 = fx(x, params)
        k2 = fx(x + (ts/2)*k1, params)
        k3 = fx(x + (ts/2)*k2, params)
        k4 = fx(x + ts*k3, params)

        states[:, t+1] = x + (ts/6)*(k1 + 2*k2 + 2*k3 + k4)

    return states


def sim_discrete(f, x0, ts, num_steps, params=None):
    """
    Simulate a discrete-time system.

    Args:
        f: Callable function of the form f(x, params) defining the discrete map.
        x0: Initial state as a torch tensor of shape (n,).
        ts: Sample time (float). For discrete maps, this is assumed to be 1 and is not used.
        num_steps: Number of simulation steps (int).
        params: Optional parameters to pass to f (default: None).

    Returns:
        states: Tensor of shape (n, num_steps) containing the simulated states.
    """
    n = x0.shape[0]
    states = torch.zeros((n, num_steps), dtype=torch.float64)
    states[:, 0] = x0

    for t in range(num_steps - 1):
        # For discrete maps, the next state is simply given by applying f to the current state.
        states[:, t+1] = f(states[:, t], params)

    return states


def sim_LTI(x0, A, C, num_steps, ts=0.01, x0cv=None):
    # Forward stepping simulation of linear time-invariant system
    # Continuous time simulation by default
    # If A and C are discrete-time matrices, set ts=None
    # Expect x0cv to be a tensor of shape (n,n)
    m, n = C.shape  # n = number of states, m = number of outputs

    x = torch.zeros((n, num_steps), dtype=x0.dtype, device=x0.device)
    y = torch.zeros((m, num_steps), dtype=x0.dtype, device=x0.device)
    A, C = A.to(device=x0.device), C.to(device=x0.device)

    if x0.dim() == 1:
        x[:, 0] = x0
        y[:, 0] = C @ x0
    elif x0.dim() == 2:
        x[:, 0] = x0[:, 0]
        y[:, 0] = C @ x0[:, 0]
    else:
        raise ValueError('Expected x0 to be a 2D tensor')

    if not ts is None:
        Ad = torch.linalg.matrix_exp(A * ts)
    else:
        Ad = A

    if x0cv is None:
        for k in range(num_steps-1):
            x[:, k+1] = Ad @ x[:, k]
            y[:, k] = C @ x[:, k]

        y[:, -1] = C @ x[:, -1]

        return x, y
    else:
        if x0cv.shape[0] is not n or x0cv.shape[1] is not n:
            raise ValueError('Expected x0cv to be a 2D tensor of shape (n,n)')

        xcv = torch.zeros((n, n, num_steps), dtype=x0.dtype, device=x0.device)
        ycv = torch.zeros((m, m, num_steps), dtype=x0.dtype, device=x0.device)
        xcv[:, :, 0] = x0cv

        for k in range(num_steps - 1):
            x[:, k+1] = Ad @ x[:, k]
            y[:, k] = C @ x[:, k]
            xcv[:, :, k+1] = Ad @ xcv[:, :, k] @ Ad.T
            ycv[:, :, k] = C @ xcv[:, :, k] @ C.T

        y[:, -1] = C @ x[:, -1]
        ycv[:, :, k] = C @ xcv[:, :, -1] @ C.T

        return x, xcv, y, ycv
