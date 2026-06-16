"""
sei_lorenz_dynamics.py
======================

SEI 形貌不稳定性 Lorenz 降维模型。

融合种子项目：
    092_bioconvection_ode：生物对流 Lorenz 降维系统

科学背景：
    在 SEI 生长过程中，界面可能出现形貌不稳定性
    （枝晶、苔藓状锂沉积）。参考 Avramenko et al. (2023)
    将偏微分方程系统降维为 Lorenz 型常微分方程组：

    dX/dt = Sc (Y - X)
    dY/dt = Ra X + X Z - Y
    dZ/dt = -X Y - b Z

    物理含义：
        X — 界面扰动幅值（浓度不均匀性）
        Y — 溶质通量偏差
        Z — 温度/电位扰动
        Sc — 有效 Schmidt 数
        Ra — 有效 Rayleigh 数（控制参数）
        b — 几何因子

作者: DA-Synthesis
"""

import math
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


# ============================================================
#  Lorenz 型 ODE 右端函数（参考 bioconvection_deriv.m）
# ============================================================

def lorenz_rhs(t, xyz, sc=None, ra=None, b=None):
    """
    Lorenz 型 ODE 系统的右端函数（参考 bioconvection_deriv.m）。

    数学形式：
        dX/dt = Sc (Y - X)
        dY/dt = Ra X + X Z - Y
        dZ/dt = -X Y - b Z

    Parameters
    ----------
    t : float
        时间（未显式使用，autonomous 系统）。
    xyz : list[float]
        状态向量 [X, Y, Z]。
    sc : float or None
        Schmidt 数，None 使用默认值。
    ra : float or None
        Rayleigh 数。
    b : float or None
        几何因子。

    Returns
    -------
    dxyzdt : list[float]
        右端函数值。
    """
    if sc is None:
        sc = P.LORENZ_SC_SEI
    if ra is None:
        ra = P.LORENZ_RAYLEIGH_SEI
    if b is None:
        b = P.LORENZ_B_SEI

    x, y, z = xyz

    dxdt = sc * (y - x)
    dydt = ra * x + x * z - y
    dzdt = -x * y - b * z

    return [dxdt, dydt, dzdt]


def lorenz_jacobian(xyz, sc=None, ra=None, b=None):
    """
    Lorenz 系统的 Jacobian 矩阵（用于隐式方法和稳定性分析）。

    J = [[-Sc,  Sc,   0  ],
         [Ra+Z, -1,   X  ],
         [-Y,   -X,  -b  ]]

    Parameters
    ----------
    xyz : list[float]
        状态向量。
    sc, ra, b : float or None
        参数。

    Returns
    -------
    J : list[list[float]]
        3×3 Jacobian 矩阵。
    """
    if sc is None:
        sc = P.LORENZ_SC_SEI
    if ra is None:
        ra = P.LORENZ_RAYLEIGH_SEI
    if b is None:
        b = P.LORENZ_B_SEI

    x, y, z = xyz

    J = [
        [-sc, sc, 0.0],
        [ra + z, -1.0, x],
        [-y, -x, -b],
    ]
    return J


# ============================================================
#  RK4 时间积分器
# ============================================================

def rk4_step(f, t, y, dt, *args):
    """
    经典四阶 Runge-Kutta 单步积分。

    数学形式：
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt k1/2)
        k3 = f(t + dt/2, y + dt k2/2)
        k4 = f(t + dt, y + dt k3)
        y(t+dt) = y(t) + (dt/6)(k1 + 2k2 + 2k3 + k4)

    Parameters
    ----------
    f : callable
        右端函数 f(t, y, *args)。
    t : float
        当前时间。
    y : list[float]
        当前状态。
    dt : float
        时间步长。
    *args :
        传递给 f 的额外参数。

    Returns
    -------
    y_new : list[float]
        新状态。
    """
    n = len(y)

    k1 = f(t, y, *args)
    y2 = [y[i] + 0.5 * dt * k1[i] for i in range(n)]
    k2 = f(t + 0.5 * dt, y2, *args)
    y3 = [y[i] + 0.5 * dt * k2[i] for i in range(n)]
    k3 = f(t + 0.5 * dt, y3, *args)
    y4 = [y[i] + dt * k3[i] for i in range(n)]
    k4 = f(t + dt, y4, *args)

    y_new = [y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
             for i in range(n)]
    return y_new


# ============================================================
#  Lorenz 系统动力学分析
# ============================================================

def find_fixed_points(sc=None, ra=None, b=None):
    """
    求 Lorenz 系统的不动点。

    不动点方程：
        0 = Sc(Y - X)  =>  Y = X
        0 = Ra X + X Z - Y  =>  (Ra - 1) X + X Z = 0
        0 = -X Y - b Z  =>  Z = -X^2 / b

    解：
        平凡不动点：(0, 0, 0)
        非平凡不动点（Ra > 1）：
            X = Y = ±sqrt(b(Ra-1)), Z = -(Ra-1)

    Parameters
    ----------
    sc, ra, b : float or None
        参数。

    Returns
    -------
    fixed_points : list[list[float]]
        不动点列表。
    """
    if ra is None:
        ra = P.LORENZ_RAYLEIGH_SEI
    if b is None:
        b = P.LORENZ_B_SEI

    fixed_points = [[0.0, 0.0, 0.0]]  # 平凡不动点

    if ra > 1.0:
        x_val = math.sqrt(b * (ra - 1.0))
        z_val = -(ra - 1.0)
        fixed_points.append([x_val, x_val, z_val])
        fixed_points.append([-x_val, -x_val, z_val])

    return fixed_points


def classify_fixed_point(xyz, sc=None, ra=None, b=None):
    """
    通过 Jacobian 本征值分类不动点的稳定性。

    Parameters
    ----------
    xyz : list[float]
        不动点。
    sc, ra, b : float or None
        参数。

    Returns
    -------
    dict
        分类结果。
    """
    J = lorenz_jacobian(xyz, sc, ra, b)

    # 3×3 矩阵特征方程：λ^3 + p λ^2 + q λ + r = 0
    tr = J[0][0] + J[1][1] + J[2][2]  # trace
    # 使用简化的本征值估计
    det_J = (J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
             - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
             + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]))

    # 简化稳定性判据
    if tr < 0 and det_J > 0:
        stability = "likely_stable"
    elif tr > 0:
        stability = "unstable"
    else:
        stability = "marginal"

    return {
        "trace": tr,
        "determinant": det_J,
        "stability": stability,
    }


def run_lorenz_trajectory(xyz0=None, n_steps=500, dt=None):
    """
    运行 Lorenz 系统时间轨迹。

    Parameters
    ----------
    xyz0 : list[float] or None
        初始条件。
    n_steps : int
        时间步数。
    dt : float or None
        时间步长。

    Returns
    -------
    dict
        轨迹数据。
    """
    if xyz0 is None:
        xyz0 = [1.0, 1.0, 1.0]
    if dt is None:
        dt = 0.01  # Lorenz 系统典型步长

    trajectory = [list(xyz0)]
    xyz = list(xyz0)
    t = 0.0

    for step in range(n_steps):
        xyz = rk4_step(lorenz_rhs, t, xyz, dt)
        # 数值安全：防止发散到无穷
        for i in range(3):
            if abs(xyz[i]) > 1.0e6:
                xyz[i] = math.copysign(1.0e6, xyz[i])
        trajectory.append(list(xyz))
        t += dt

    fixed_points = find_fixed_points()
    classifications = []
    for fp in fixed_points:
        cls = classify_fixed_point(fp)
        classifications.append({"point": fp, "classification": cls})

    return {
        "trajectory": trajectory,
        "n_steps": n_steps,
        "dt": dt,
        "fixed_points": classifications,
        "final_state": trajectory[-1],
    }


# ============================================================
#  分岔分析
# ============================================================

def bifurcation_scan(ra_values, sc=None, b=None):
    """
    对 Rayleigh 数进行分岔扫描。

    Parameters
    ----------
    ra_values : list[float]
        Ra 值列表。
    sc : float or None
        Schmidt 数。
    b : float or None
        几何因子。

    Returns
    -------
    results : list[dict]
        每个 Ra 值的分析结果。
    """
    results = []
    for ra in ra_values:
        fps = find_fixed_points(sc=sc, ra=ra, b=b)
        stable_fps = []
        for fp in fps:
            cls = classify_fixed_point(fp, sc=sc, ra=ra, b=b)
            if cls['stability'] == 'likely_stable':
                stable_fps.append(fp)

        results.append({
            "ra": ra,
            "n_fixed_points": len(fps),
            "n_stable": len(stable_fps),
            "regime": ("subcritical" if ra < 1.0
                       else "supercritical" if len(stable_fps) > 0
                       else "chaotic_candidate"),
        })
    return results


if __name__ == "__main__":
    result = run_lorenz_trajectory(n_steps=200)
    print(f"[sei_lorenz] Lorenz 轨迹模拟")
    print(f"  初始状态: [1.0, 1.0, 1.0]")
    print(f"  最终状态: [{result['final_state'][0]:.4f}, "
          f"{result['final_state'][1]:.4f}, {result['final_state'][2]:.4f}]")

    print(f"\n  不动点分析:")
    for fp_info in result['fixed_points']:
        fp = fp_info['point']
        cls = fp_info['classification']
        print(f"    ({fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f}) -> "
              f"trace={cls['trace']:.3f}, {cls['stability']}")

    # 分岔扫描
    ra_vals = [0.5, 1.0, 5.0, 10.0, 25.0, 49.6, 100.0]
    bifur = bifurcation_scan(ra_vals)
    print(f"\n  分岔扫描:")
    for r in bifur:
        print(f"    Ra={r['ra']:6.1f}: {r['n_fixed_points']} 不动点, "
              f"{r['n_stable']} 稳定, 状态={r['regime']}")
