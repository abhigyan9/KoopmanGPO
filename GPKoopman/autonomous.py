# models and equations
import torch


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
    if x[0] < 0.:
        dx0 = params[2] * x[0] - params[2] * params[3] - x[0]
    else:
        dx0 = params[1] * x[0] + params[0] - params[1] * x[0] - x[0]

    return torch.tensor([dx0], dtype=torch.float64)


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


def sim_LTI(x0, A, C, num_steps, ts=0.01, x0cv=None):
    # Forward stepping simulation of linear time-invariant system
    # Continuous time simulation by default
    # If A and C are discrete-time matrices, set ts=None
    # Expect x0cv to be a tensor of shape (n,n)
    m, n = C.shape  # n = number of states, m = number of outputs

    x = torch.zeros((n, num_steps))
    y = torch.zeros((m, num_steps))

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

        xcv = torch.zeros((n, n, num_steps))
        ycv = torch.zeros((m, m, num_steps))
        xcv[:, :, 0] = x0cv

        for k in range(num_steps - 1):
            x[:, k+1] = Ad @ x[:, k]
            y[:, k] = C @ x[:, k]
            xcv[:, :, k+1] = Ad @ xcv[:, :, k] @ Ad.T
            ycv[:, :, k] = C @ xcv[:, :, k] @ C.T

        y[:, -1] = C @ x[:, -1]
        ycv[:, :, k] = C @ xcv[:, :, -1] @ C.T

        return x, xcv, y, ycv
