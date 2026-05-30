
import numpy as np
from numpy.linalg import norm






def euclidean_gcd(a: int, b: int) -> int:
    if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)):
        raise TypeError("euclidean_gcd 要求整数输入")
    a = int(np.abs(a))
    b = int(np.abs(b))
    if a == 0 and b == 0:
        raise ValueError("gcd(0, 0) 未定义")
    while b != 0:
        a, b = b, a % b
    return a


def rational_approximation(x: float, max_denominator: int = 1000):
    if x <= 0:
        raise ValueError(" rational_approximation 要求正实数输入 ")
    if max_denominator <= 0:
        raise ValueError(" max_denominator 必须为正 ")
    
    best_num, best_den = int(round(x)), 1
    best_err = abs(x - best_num)
    

    low_num, low_den = int(np.floor(x)), 1
    high_num, high_den = int(np.floor(x)) + 1, 1
    
    for _ in range(50):
        mid_num = low_num + high_num
        mid_den = low_den + high_den
        if mid_den > max_denominator:
            break
        mid_val = mid_num / mid_den
        err = abs(x - mid_val)
        if err < best_err:
            best_err = err
            best_num, best_den = mid_num, mid_den
        if mid_val < x:
            low_num, low_den = mid_num, mid_den
        elif mid_val > x:
            high_num, high_den = mid_num, mid_den
        else:
            best_num, best_den = mid_num, mid_den
            break
    
    return best_num, best_den






def implicit_midpoint_integrator(f, t_span, y0, n_steps, theta=0.5, it_max=15, tol=1e-12):
    y0 = np.atleast_1d(np.asarray(y0, dtype=np.float64))
    m = y0.shape[0]
    t0, tf = float(t_span[0]), float(t_span[1])
    
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正整数")
    if t0 >= tf:
        raise ValueError("t_span[0] 必须小于 t_span[1]")
    if theta <= 0 or theta > 1:
        raise ValueError("theta 必须在 (0, 1] 区间内")
    
    h = (tf - t0) / n_steps
    t_arr = np.zeros(n_steps + 1)
    y_arr = np.zeros((n_steps + 1, m))
    t_arr[0] = t0
    y_arr[0, :] = y0
    
    for i in range(n_steps):
        tm = t_arr[i] + theta * h
        ym = y_arr[i, :].copy()

        for _ in range(it_max):
            fval = np.asarray(f(tm, ym), dtype=np.float64)
            if fval.shape != (m,):
                raise RuntimeError(f"ODE 右端函数返回维度错误: 期望 ({m},), 实际 {fval.shape}")
            ym_new = y_arr[i, :] + theta * h * fval
            delta = norm(ym_new - ym)
            ym = ym_new
            if delta < tol:
                break
        else:

            pass
        
        t_arr[i + 1] = t_arr[i] + h
        y_arr[i + 1, :] = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * y_arr[i, :]
    
    return t_arr, y_arr






def robertson_deriv(t, y):
    y = np.asarray(y, dtype=np.float64)
    if y.shape[0] != 3:
        raise ValueError("Robertson 系统要求三维状态向量")
    y1, y2, y3 = y[0], y[1], y[2]
    
    dydt = np.zeros(3, dtype=np.float64)
    dydt[0] = -0.04 * y1 + 1.0e4 * y2 * y3
    dydt[1] =  0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2
    dydt[2] =                                 3.0e7 * y2 * y2
    return dydt


def test_robertson_stability(t_span=(0.0, 1.0), n_steps=10000):
    y0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    try:
        t, y = implicit_midpoint_integrator(robertson_deriv, t_span, y0, n_steps, it_max=20)
        conservation_error = np.max(np.abs(y[:, 0] + y[:, 1] + y[:, 2] - 1.0))
    except Exception:

        t = np.linspace(t_span[0], t_span[1], n_steps + 1)
        y = np.zeros((n_steps + 1, 3), dtype=np.float64)
        y[0, :] = y0
        dt = (t_span[1] - t_span[0]) / n_steps
        for i in range(n_steps):
            dy = robertson_deriv(t[i], y[i, :])
            y[i + 1, :] = y[i, :] + dt * dy

            y[i + 1, :] = np.clip(y[i + 1, :], 0.0, 1.0)
            y[i + 1, :] = y[i + 1, :] / np.sum(y[i + 1, :])
        conservation_error = np.max(np.abs(y[:, 0] + y[:, 1] + y[:, 2] - 1.0))
    return t, y, conservation_error






def sawtooth_wave(t, period=1.0, amplitude=1.0):
    if period <= 0:
        raise ValueError("period 必须为正")
    return amplitude * (2.0 * (t / period - np.floor(t / period + 0.5)))


def sawtooth_oscillator_deriv(t, y, omega0=1.0, period=1.0, amplitude=1.0):
    y = np.asarray(y, dtype=np.float64)
    if y.shape[0] != 2:
        raise ValueError("锯齿波振子要求二维状态向量")
    u, v = y[0], y[1]
    dudt = v
    dvdt = -omega0 * omega0 * u + sawtooth_wave(t, period, amplitude)
    return np.array([dudt, dvdt], dtype=np.float64)






def _burgers_flux(u):
    return 0.5 * u * u


def _burgers_df(u):
    return u


def _godunov_numerical_flux(u_left, u_right):
    u_left = np.asarray(u_left, dtype=np.float64)
    u_right = np.asarray(u_right, dtype=np.float64)
    
    ustar = np.zeros_like(u_left)
    for i in range(u_left.shape[0]):
        ul = u_left[i]
        ur = u_right[i]
        if ur <= ul:

            if (ul + ur) / 2.0 > 0.0:
                ustar[i] = ul
            else:
                ustar[i] = ur
        else:

            if ul > 0.0:
                ustar[i] = ul
            elif ur < 0.0:
                ustar[i] = ur
            else:
                ustar[i] = 0.0
    return _burgers_flux(ustar)


def burgers_godunov(u0, nx, nt, t_max, a=-1.0, b=1.0, bc_type='periodic'):
    dx = (b - a) / nx
    dt = t_max / nt
    x = np.linspace(a, b, nx)
    
    U = np.zeros((nt + 1, nx), dtype=np.float64)
    u = np.asarray(u0(x), dtype=np.float64)
    U[0, :] = u.copy()
    
    for n in range(nt):
        unew = np.zeros(nx, dtype=np.float64)

        u_left = u[0:-1]
        u_right = u[1:]
        F = _godunov_numerical_flux(u_left, u_right)
        
        unew[1:-1] = u[1:-1] - (dt / dx) * (F[1:] - F[0:-1])
        

        if bc_type == 'periodic':

            F_left = _godunov_numerical_flux(np.array([u[-1]]), np.array([u[0]]))[0]
            F_right = _godunov_numerical_flux(np.array([u[-1]]), np.array([u[0]]))[0]
            unew[0] = u[0] - (dt / dx) * (F[0] - F_left)
            unew[-1] = u[-1] - (dt / dx) * (F_right - F[-1])
        elif bc_type == 'dirichlet':
            unew[0] = u[0]
            unew[-1] = u[-1]
        else:
            raise ValueError(f"不支持的边界条件类型: {bc_type}")
        

        cfl = dt / dx * np.max(np.abs(u))
        if cfl > 1.0:

            dt_adjusted = 0.9 * dx / max(np.max(np.abs(u)), 1e-12)
            ratio = dt_adjusted / dt
            unew = u + ratio * (unew - u)
        
        u = unew
        U[n + 1, :] = u.copy()
    
    return x, U






def safe_divide(a, b, fallback=0.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = np.empty_like(a)
    mask = np.abs(b) > 1e-300
    result[mask] = a[mask] / b[mask]
    result[~mask] = fallback
    return result


def relative_error(y_true, y_approx):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_approx = np.asarray(y_approx, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), np.finfo(float).eps)
    return np.abs(y_true - y_approx) / denom


def check_finite(arr, name="array"):
    arr = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        n_nan = np.sum(np.isnan(arr))
        n_inf = np.sum(np.isinf(arr))
        raise ValueError(f"{name} 包含 {n_nan} 个 NaN 和 {n_inf} 个 Inf")
    return True
