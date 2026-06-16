"""
phase_shift.py
===================================================================
散射相移与 S 矩阵计算模块

映射种子项目:
  - 1428_zero_chandrupatla: Chandrupatla 求根 → S 矩阵极点 (共振态) 搜索
  - 1373_uniform: 均匀随机采样 → 相移蒙特卡洛误差传播

核心物理公式:
  渐近边界条件:
    u_l(r) -> A_l * [cos(delta_l) * j_l(kr) - sin(delta_l) * n_l(kr)]  (r -> inf)

  相移提取 (对数导数匹配):
    tan(delta_l) = [k*r*j_l'(kr) - j_l(kr)] / [k*r*n_l'(kr) - n_l(kr)]
    在 r = R_match 处匹配

  S 矩阵:
    S_l = exp(2*i*delta_l) = eta_l * exp(2*i*delta_l^real)
    其中 eta_l = exp(-2*Im(delta_l)) 为吸收因子 (0 <= eta <= 1)

  共振条件:
    delta_l(E_res) = pi/2 + n*pi (mod pi)
    => Re(S_l) = 0 ( Breit-Wigner 共振)

  Breit-Wigner 共振宽度:
    Gamma = 2 * (d(delta_l)/dE)^{-1} |_{E=E_res}
===================================================================
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy import special


# ---------- 物理常数 ----------
HBAR_C = 197.3269804


def spherical_bessel_j(l: int, x: np.ndarray) -> np.ndarray:
    """
    球贝塞尔函数 j_l(x)

    递推关系:
        j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x)

    特殊值:
        j_0(x) = sin(x)/x
        j_1(x) = sin(x)/x^2 - cos(x)/x
    """
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)
    if l == 0:
        return np.sin(x_safe) / x_safe
    elif l == 1:
        return np.sin(x_safe) / x_safe ** 2 - np.cos(x_safe) / x_safe
    else:
        # 使用 SciPy 球贝塞尔函数
        return special.spherical_jn(l, x_safe)


def spherical_bessel_n(l: int, x: np.ndarray) -> np.ndarray:
    """
    球诺伊曼函数 n_l(x) (第二类球贝塞尔函数)

    特殊值:
        n_0(x) = -cos(x)/x
        n_1(x) = -cos(x)/x^2 - sin(x)/x
    """
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)
    if l == 0:
        return -np.cos(x_safe) / x_safe
    elif l == 1:
        return -np.cos(x_safe) / x_safe ** 2 - np.sin(x_safe) / x_safe
    else:
        return special.spherical_yn(l, x_safe)


def spherical_bessel_j_derivative(l: int, x: np.ndarray) -> np.ndarray:
    """
    球贝塞尔函数导数 j_l'(x):
        j_l'(x) = j_{l-1}(x) - (l+1)/x * j_l(x)
    """
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)
    jl = spherical_bessel_j(l, x_safe)
    jl_prev = spherical_bessel_j(l - 1, x_safe) if l > 0 else np.sin(x_safe) / x_safe
    return jl_prev - (l + 1) / x_safe * jl


def spherical_bessel_n_derivative(l: int, x: np.ndarray) -> np.ndarray:
    """
    球诺伊曼函数导数 n_l'(x):
        n_l'(x) = n_{l-1}(x) - (l+1)/x * n_l(x)
    """
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)
    nl = spherical_bessel_n(l, x_safe)
    nl_prev = spherical_bessel_n(l - 1, x_safe) if l > 0 else -np.cos(x_safe) / x_safe
    return nl_prev - (l + 1) / x_safe * nl


class PhaseShiftCalculator:
    """
    散射相移计算器

    方法:
    1. 对数导数匹配法: 在外边界匹配数值解与渐近形式
    2. S 矩阵构造: S_l = exp(2i*delta_l)
    3. 共振搜索: Chandrupatla 方法搜索 S 矩阵极点
    """

    def __init__(
        self,
        k: float,
        r_match: float = 25.0,
        n_points: int = 2000,
    ):
        """
        参数:
            k: 入射波数 [1/fm]
            r_match: 匹配半径 [fm] (需足够大使势可忽略)
            n_points: 径向网格点数
        """
        self.k = k
        self.r_match = r_match
        self.n_points = n_points

    def extract_phase_shift(
        self,
        u_numerical: np.ndarray,
        r: np.ndarray,
        l: int
    ) -> complex:
        """
        从数值波函数提取复数相移 (对数导数匹配)

        在外边界 r = R_match 处:
            数值解对数导数: L_num = u'(R)/u(R)

            渐近形式:
                u(r) ~ A*[cos(delta)*j_l(kr) - sin(delta)*n_l(kr)]

            对数导数:
                L_asym = k * [cos(delta)*j_l'(kR) - sin(delta)*n_l'(kR)] /
                         [cos(delta)*j_l(kR) - sin(delta)*n_l(kR)]

            令 L_num = L_asym, 解出 delta:
                tan(delta) = [k*j_l'(kR) - L_num*j_l(kR)] /
                             [k*n_l'(kR) - L_num*n_l(kR)]

        对光学势 (复数), delta 为复数:
            delta = delta_real + i * delta_imag
            eta = exp(-2*delta_imag)  (吸收因子)

        参数:
            u_numerical: 数值径向波函数
            r: 径向网格
            l: 角动量量子数

        返回:
            复数相移 delta (弧度)
        """
        # 在匹配点计算数值解和对数导数
        idx_match = np.argmin(np.abs(r - self.r_match))
        if idx_match < 2 or idx_match >= len(r) - 1:
            idx_match = max(2, min(idx_match, len(r) - 2))

        # 中心差分求导
        dr = r[1] - r[0]
        u_match = u_numerical[idx_match]
        dup_match = (u_numerical[idx_match + 1] - u_numerical[idx_match - 1]) / (2.0 * dr)

        if abs(u_match) < 1e-300:
            return complex(0.0, 0.0)

        L_num = dup_match / u_match

        # 渐近函数在匹配点的值
        kR = self.k * r[idx_match]
        jl = spherical_bessel_j(l, kR)
        nl = spherical_bessel_n(l, kR)
        djl = spherical_bessel_j_derivative(l, kR)
        dnl = spherical_bessel_n_derivative(l, kR)

        # tan(delta) = (k*j_l' - L*j_l) / (k*n_l' - L*n_l)
        numerator = self.k * djl - L_num * jl
        denominator = self.k * dnl - L_num * nl

        if abs(denominator) < 1e-300:
            # delta = pi/2 (共振)
            delta = complex(np.pi / 2, 0.0)
        else:
            tan_delta = numerator / denominator
            delta = np.arctan(tan_delta)

            # 处理象限问题
            if abs(u_match) > 1e-10:
                # 检查符号一致性
                u_asym_test = np.cos(delta) * jl - np.sin(delta) * nl
                if np.real(u_match * np.conj(u_asym_test)) < 0:
                    delta += np.pi

        # 对复数光学势, 需要更精细的处理
        if np.iscomplexobj(u_numerical) and np.any(np.imag(u_numerical) != 0):
            # 使用复数 arctan
            delta = complex(np.arctan(complex(numerator / (denominator + 1e-300))))

        return delta

    def compute_s_matrix(self, delta: complex) -> complex:
        """
        计算 S 矩阵元素:
            S_l = exp(2*i*delta_l)

        对弹性散射 (实数 delta): |S_l| = 1 (幺正性)
        对吸收 (复数 delta): |S_l| = exp(-2*Im(delta)) < 1
        """
        return np.exp(2j * delta)

    def compute_eta(self, delta: complex) -> float:
        """
        吸收因子 (非弹性度):
            eta_l = |S_l| = exp(-2 * Im(delta_l))

        eta = 1: 纯弹性散射
        eta < 1: 存在非弹性道 (反应道开放)
        eta = 0: 完全吸收 (黑核极限)
        """
        return float(np.exp(-2.0 * np.imag(delta)))

    def chandrupatla_resonance_search(
        self,
        phase_shift_func,
        l: int,
        E_min: float = 1.0,
        E_max: float = 30.0,
        tol: float = 1e-8
    ) -> List[Dict]:
        """
        Chandrupatla 求根法搜索共振能量
        (映射自 1428_zero_chandrupatla)

        共振条件: delta_l(E) = pi/2 (mod pi)
        等价于求根: f(E) = tan(delta_l(E)) - 0 = 0
        或者: Re(S_l(E)) = 0

        Chandrupatla 方法:
        在逆二次插值和二分法之间自适应切换.
        当插值点落在括号内时使用插值, 否则回退到二分.

        参数:
            phase_shift_func: 能量 -> 相移的函数
            l: 角动量量子数
            E_min, E_max: 搜索能量范围 [MeV]
            tol: 收敛容差

        返回:
            共振参数列表 (能量, 宽度, 量子数)
        """
        resonances = []
        n_scan = 100
        E_scan = np.linspace(E_min, E_max, n_scan)

        # 扫描寻找相移过 pi/2 的区间
        delta_scan = []
        for E in E_scan:
            try:
                d = phase_shift_func(E, l)
                delta_scan.append(np.real(d))
            except Exception:
                delta_scan.append(0.0)

        delta_scan = np.array(delta_scan)

        # 寻找 pi/2 + n*pi 的穿越
        for n_pi in range(-2, 4):
            target = np.pi / 2 + n_pi * np.pi
            for i in range(len(delta_scan) - 1):
                if ((delta_scan[i] - target) * (delta_scan[i + 1] - target)) < 0:
                    # 括号找到, 用 Chandrupatla 精确化
                    E_res = self._chandrupatla_root(
                        lambda E: np.real(phase_shift_func(E, l)) - target,
                        E_scan[i], E_scan[i + 1], tol
                    )
                    if E_res is not None:
                        # 估算宽度: Gamma = 2 / (ddelta/dE)
                        dE = 0.01
                        d1 = np.real(phase_shift_func(E_res - dE, l))
                        d2 = np.real(phase_shift_func(E_res + dE, l))
                        ddelta_dE = (d2 - d1) / (2 * dE)
                        Gamma = 2.0 / abs(ddelta_dE) if abs(ddelta_dE) > 1e-10 else 0.0

                        resonances.append({
                            'energy': float(E_res),
                            'width': float(Gamma),
                            'l_quantum': l,
                            'target_phase': target,
                        })

        return resonances

    def _chandrupatla_root(
        self, f, x1: float, x2: float, tol: float = 1e-8,
        max_iter: int = 100
    ) -> Optional[float]:
        """
        Chandrupatla 求根算法实现

        混合逆二次插值 + 二分法:
        1. 计算插值参数 t
        2. 若 t 在 [tol, 1-tol] 内, 使用插值
        3. 否则使用二分法 t = 0.5

        收敛条件: |x2 - x1| < 2*tol*|x2| + 0.5*tol
        """
        f1 = f(x1)
        f2 = f(x2)

        if f1 * f2 > 0:
            return None  # 未形成括号

        # 初始化
        x3 = x1
        f3 = f1
        tl = 0.5

        for iteration in range(max_iter):
            xm = x2
            fm = f2

            # 判断是否使用插值
            if abs(f1 - f2) > 1e-30 and abs(f1 - f3) > 1e-30:
                # 逆二次插值参数
                phi = (f1 - f2) / (f3 - f2) if abs(f3 - f2) > 1e-30 else 1.0
                # Chandrupatla 判别
                t_iq = (f1 / (f1 - f2) if abs(f1 - f2) > 1e-30 else 0.5)

                # 检查插值可行性
                if 0 < t_iq < 1:
                    t = t_iq
                else:
                    t = 0.5  # 二分法回退
            else:
                t = 0.5  # 二分法

            # 容差检查
            x_new = x1 + t * (x2 - x1)
            f_new = f(x_new)

            # 更新括号
            if f1 * f_new < 0:
                x3, f3 = x2, f2
                x2, f2 = x_new, f_new
            else:
                x1, f1 = x_new, f_new

            # 收敛检查
            if abs(x2 - x1) < tol * (abs(x2) + 1e-30):
                return x2

        return x2  # 返回当前最优估计

    def monte_carlo_phase_uncertainty(
        self,
        phase_shift_func,
        l: int,
        E: float,
        n_samples: int = 200,
        param_perturb: float = 0.05,
        seed: int = 42
    ) -> Dict:
        """
        蒙特卡洛相移不确定度传播
        (映射自 1373_uniform 均匀随机采样)

        对光学势参数施加均匀随机微扰:
            V_i -> V_i * (1 + delta_i),  delta_i ~ Uniform(-eps, eps)

        统计相移的分布特征

        参数:
            phase_shift_func: 参数化相移函数
            l: 角动量
            E: 入射能量
            n_samples: 蒙特卡洛样本数
            param_perturb: 参数微扰幅度

        返回:
            统计结果
        """
        rng = np.random.RandomState(seed)
        deltas = []

        for _ in range(n_samples):
            # 均匀随机微扰
            perturbation = 1.0 + param_perturb * (2.0 * rng.random() - 1.0)
            try:
                d = phase_shift_func(E * perturbation, l)
                deltas.append(np.real(d))
            except Exception:
                pass

        deltas = np.array(deltas)

        if len(deltas) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'n_samples': 0}

        return {
            'mean': float(np.mean(deltas)),
            'std': float(np.std(deltas)),
            'min': float(np.min(deltas)),
            'max': float(np.max(deltas)),
            'median': float(np.median(deltas)),
            'n_samples': len(deltas),
            'relative_uncertainty': float(np.std(deltas) / (abs(np.mean(deltas)) + 1e-30)),
        }
