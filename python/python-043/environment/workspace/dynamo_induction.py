"""
地核发电机感应方程求解器 (dynamo_induction.py)
=================================================
本模块实现运动学 α-Ω Dynamo 模型的核心物理方程：

  感应方程（磁场演化）:
    ∂B/∂t = ∇×(u×B) + η∇²B + α(∇×B)

  其中:
    B  : 磁场 (T)
    u  : 速度场 (m/s)，参数化为 u = u_Ω + u_α
    η  : 磁扩散系数 (m²/s)
    α  : α 效应参数化系数 (m/s)

  采用环向-极向分解 (toroidal-poloidal decomposition):
    B = ∇×(T r̂) + ∇×∇×(P r̂)

  其中 T(r,θ,φ,t) 和 P(r,θ,φ,t) 为标量函数，自动满足 ∇·B = 0。

  在球谐展开后，每个 (l,m) 模式满足径向方程组：

    ∂T_lm/∂t = η [ (1/r²) ∂/∂r(r² ∂T_lm/∂r) - l(l+1)/r² T_lm ] + S_T(l,m)
    ∂P_lm/∂t = η [ (1/r²) ∂/∂r(r² ∂P_lm/∂r) - l(l+1)/r² P_lm ] + S_P(l,m)

  源项:
    S_T = α * l(l+1)/r² * P_lm               (α 效应产生环向场)
    S_P = 耦合项（Ω 效应将环向场转化为极向场）

本模块提供：
  - 速度场参数化（差速自转 + α 效应湍流）
  - 感应方程右端项计算
  - 完整时间演化循环
"""

import numpy as np
from typing import Dict, Tuple, List
from radial_solver import evolve_radial_modes, alpha_effect_source, omega_effect_source
from adaptive_rk import rk45_adaptive
from special_functions import safe_div


# ---------------------------------------------------------------------------
# 1. 速度场参数化
# ---------------------------------------------------------------------------
def differential_rotation_profile(r: np.ndarray, r_icb: float, r_cmb: float,
                                   Omega0: float, shear_strength: float) -> np.ndarray:
    """
    差速自转角速度剖面：
      Ω(r) = Ω0 * [1 - shear * (r - r_icb)/(r_cmb - r_icb)]

    这是地核发电机 Ω 效应的核心驱动力。外核外层自转较慢，
    将极向磁场剪切为环向磁场。
    """
    d = r_cmb - r_icb
    if d <= 0.0:
        return np.zeros_like(r)
    x = (r - r_icb) / d
    return Omega0 * (1.0 - shear_strength * x)


def alpha_effect_profile(r: np.ndarray, r_icb: float, r_cmb: float,
                          alpha0: float) -> np.ndarray:
    """
    α 效应参数化剖面。α 效应由小尺度湍流驱动的螺旋度产生。
    在核幔边界附近（对流最强处）最大，在 ICB 和 CMB 处为零。

    公式:
      α(r) = alpha0 * sin(π * (r - r_icb) / (r_cmb - r_icb))
    """
    d = r_cmb - r_icb
    if d <= 0.0:
        return np.zeros_like(r)
    alpha = alpha0 * np.sin(np.pi * (r - r_icb) / d)
    alpha[r <= r_icb] = 0.0
    alpha[r >= r_cmb] = 0.0
    return alpha


# ---------------------------------------------------------------------------
# 2. 感应方程右端项（径向-球谐模式空间）
# ---------------------------------------------------------------------------
def induction_rhs(state: np.ndarray, r: np.ndarray,
                  r_icb: float, r_cmb: float,
                  eta: float, alpha0: float,
                  Omega0: float, shear_strength: float,
                  mode_list: List[Tuple[int, int]]) -> np.ndarray:
    """
    计算感应方程在状态向量 state 处的右端项 d(state)/dt。

    state 编码方式:
      对于每个 (l,m) 模式，包含 n_radial 个 T_lm 值和 n_radial 个 P_lm 值。
      总长度 = len(mode_list) * 2 * n_radial

    物理过程:
      - 磁扩散（二阶径向导数 + l(l+1)/r² 衰减）
      - α 效应（极向 -> 环向）
      - Ω 效应（环向 -> 极向，通过差速自转剪切）
    """
    n_r = len(r)
    n_modes = len(mode_list)
    rhs = np.zeros_like(state)

    # 解析 state
    T_modes = {}
    P_modes = {}
    offset = 0
    for key in mode_list:
        T_modes[key] = state[offset: offset + n_r]
        offset += n_r
        P_modes[key] = state[offset: offset + n_r]
        offset += n_r

    # 预计算速度场
    Omega_profile = differential_rotation_profile(r, r_icb, r_cmb, Omega0, shear_strength)
    alpha_profile = alpha_effect_profile(r, r_icb, r_cmb, alpha0)

    # TODO(Hole_1): 实现感应方程右端项的核心物理计算循环。
    # 对 mode_list 中每个 (l,m) 模式，计算环向场 T 和极向场 P 的演化率：
    #   rhs_T = eta * (磁扩散离散) + alpha效应源项
    #   rhs_P = eta * (磁扩散离散) + Omega效应源项
    # 需包含：径向二阶差分、球谐衰减项、边界条件处理。
    # 参考: 球坐标径向扩散离散、alpha_effect_profile、Omega_profile。
    raise NotImplementedError("Hole_1: induction_rhs 核心循环待实现")

    return rhs


# ---------------------------------------------------------------------------
# 3. 状态向量编解码
# ---------------------------------------------------------------------------
def encode_state(T_modes: Dict[Tuple[int, int], np.ndarray],
                 P_modes: Dict[Tuple[int, int], np.ndarray],
                 mode_list: List[Tuple[int, int]]) -> np.ndarray:
    """将 T_modes 和 P_modes 编码为长向量。"""
    n_r = len(T_modes[mode_list[0]])
    state = np.zeros(len(mode_list) * 2 * n_r, dtype=float)
    offset = 0
    for key in mode_list:
        state[offset: offset + n_r] = T_modes[key]
        offset += n_r
        state[offset: offset + n_r] = P_modes[key]
        offset += n_r
    return state


def decode_state(state: np.ndarray,
                 mode_list: List[Tuple[int, int]],
                 n_r: int) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """将长向量解码为 T_modes 和 P_modes。"""
    T_modes = {}
    P_modes = {}
    offset = 0
    for key in mode_list:
        T_modes[key] = state[offset: offset + n_r].copy()
        offset += n_r
        P_modes[key] = state[offset: offset + n_r].copy()
        offset += n_r
    return T_modes, P_modes


# ---------------------------------------------------------------------------
# 4. 完整发电机模拟循环
# ---------------------------------------------------------------------------
def run_kinematic_dynamo(
    r: np.ndarray,
    r_icb: float,
    r_cmb: float,
    eta: float,
    alpha0: float,
    Omega0: float,
    shear_strength: float,
    l_max: int,
    t_end: float,
    dt_init: float,
    save_interval: float,
    adaptive_tol: float = 1e-6
) -> Tuple[List[float], List[Dict[Tuple[int, int], complex]], List[Dict[Tuple[int, int], complex]]]:
    """
    运行运动学地核发电机模拟。

    返回:
      times         : 保存时刻列表 (秒)
      T_coeffs_history : 每个时刻的环向场球谐系数
      P_coeffs_history : 每个时刻的极向场球谐系数

    注意：为简化计算，这里将径向网格平均值作为球谐系数输出，
    真实应用需要完整的 3D 综合变换。
    """
    n_r = len(r)
    mode_list = []
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            mode_list.append((l, m))

    n_modes = len(mode_list)

    # 初始条件：弱偶极子极向场 + 微弱环向场
    T_modes = {}
    P_modes = {}
    for key in mode_list:
        l, m = key
        # 初始极向场：集中在核幔边界附近
        P_init = np.sin(np.pi * (r - r_icb) / (r_cmb - r_icb)) * (l == 1 and m == 0)
        P_init[r <= r_icb] = 0.0
        P_init[r >= r_cmb] = 0.0
        P_modes[key] = P_init

        # 初始环向场：随机小扰动
        rng = np.random.default_rng(seed=42 + l * 100 + abs(m))
        T_modes[key] = 0.01 * rng.random(n_r) * (np.sin(np.pi * (r - r_icb) / (r_cmb - r_icb)))

    state0 = encode_state(T_modes, P_modes, mode_list)

    # 构建感应方程 RHS 函数
    def rhs_func(t, y):
        return induction_rhs(y, r, r_icb, r_cmb, eta, alpha0,
                             Omega0, shear_strength, mode_list)

    # 使用 RK45 自适应积分
    print(f"[Dynamo] Starting simulation: l_max={l_max}, modes={n_modes}, t_end={t_end/1e3/365.25/24/3600:.1f} kyrs")
    t_array, y_array, e_array = rk45_adaptive(rhs_func, (0.0, t_end), state0,
                                               dt_init=dt_init, tol=adaptive_tol)
    print(f"[Dynamo] Simulation complete: {len(t_array)} steps, final dt={t_array[-1]-t_array[-2] if len(t_array)>1 else 0:.3e} s")

    # 后处理：按 save_interval 抽取结果
    times = []
    T_history = []
    P_history = []
    next_save = 0.0

    for i in range(len(t_array)):
        if t_array[i] >= next_save or i == 0 or i == len(t_array) - 1:
            times.append(t_array[i])
            Ti, Pi = decode_state(y_array[i], mode_list, n_r)
            # 将径向平均值作为球谐系数
            T_coeffs = {key: float(np.mean(Ti[key])) + 0.0j for key in mode_list}
            P_coeffs = {key: float(np.mean(Pi[key])) + 0.0j for key in mode_list}
            T_history.append(T_coeffs)
            P_history.append(P_coeffs)
            next_save += save_interval

    return times, T_history, P_history


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    r_icb = 1221e3
    r_cmb = 3480e3
    n_r = 16
    r = np.linspace(r_icb, r_cmb, n_r)
    eta = 2.0
    alpha0 = 0.5
    Omega0 = 7.292e-5
    shear = 1.0

    mode_list = [(1, 0), (2, 0)]
    n_modes = len(mode_list)
    state = np.zeros(n_modes * 2 * n_r, dtype=float)
    state[n_r: 2 * n_r] = np.sin(np.pi * (r - r_icb) / (r_cmb - r_icb))

    rhs = induction_rhs(state, r, r_icb, r_cmb, eta, alpha0, Omega0, shear, mode_list)
    assert rhs.shape == state.shape
    assert not np.isnan(rhs).any()

    # 运行短模拟
    times, T_hist, P_hist = run_kinematic_dynamo(
        r, r_icb, r_cmb, eta, alpha0, Omega0, shear,
        l_max=2, t_end=1e4 * 365.25 * 24 * 3600, dt_init=1e3 * 365.25 * 24 * 3600,
        save_interval=5e3 * 365.25 * 24 * 3600
    )
    assert len(times) > 0
    print("dynamo_induction: self-test passed.")


if __name__ == "__main__":
    _self_test()
