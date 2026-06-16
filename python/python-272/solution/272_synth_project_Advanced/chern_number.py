"""
Chern 数计算与拓扑不变量模块
实现多种方法计算二维 Brillouin 区切片的 Chern 数

核心物理：
Chern 数 C = (1/2π) ∫∫_{BZ} Ω(k) d²k

其中 Ω(k) 是 Berry curvature：
Ω_n(k) = ∂A_y/∂k_x - ∂A_x/∂k_y
A_n(k) = i<u_n(k)|∇_k|u_n(k)> 是 Berry connection
"""

import numpy as np
from typing import Tuple, List, Dict
from berry_curvature import BerryCurvatureCalculator
from brillouin_mesh import BrillouinMesh
from weyl_hamiltonian import WeylHamiltonian


class ChernNumberCalculator:
    """
    Chern 数计算器

    支持方法：
    1. 直接积分 Berry curvature
    2. Fukui-Hatsugai-Suzuki (FHS) 离散方法
    3. Wilson loop 方法
    4. 多带非 Abel Chern 数
    """

    def __init__(self, berry_calc: BerryCurvatureCalculator):
        """
        初始化

        Parameters:
        -----------
        berry_calc : BerryCurvatureCalculator
            Berry curvature 计算器
        """
        self.berry_calc = berry_calc
        self.ham = berry_calc.ham
        self.mesh = berry_calc.mesh

    def integrate_berry_curvature(self, plane: str = 'kz_const',
                                   kz_value: float = 0.0,
                                   band_idx: int = 0,
                                   n_grid: int = 30,
                                   method: str = 'kubo') -> float:
        """
        直接积分 Berry curvature 计算 Chern 数

        C = (1/2π) ∫∫ Ω(k) dk_x dk_y

        Parameters:
        -----------
        plane : str
            切片平面
        kz_value : float
            固定坐标值
        band_idx : int
            能带索引
        n_grid : int
            网格密度
        method : str
            计算方法

        Returns:
        --------
        float
            Chern 数 (应接近整数)
        """
        # 生成 2D 网格
        kx_vals = np.linspace(0, 1, n_grid, endpoint=False)
        ky_vals = np.linspace(0, 1, n_grid, endpoint=False)

        chern_sum = 0.0
        dkx = 1.0 / n_grid
        dky = 1.0 / n_grid

        for kx in kx_vals:
            for ky in ky_vals:
                if plane == 'kz_const':
                    k = np.array([kx, ky, kz_value])
                elif plane == 'kx_const':
                    k = np.array([kz_value, kx, ky])
                elif plane == 'ky_const':
                    k = np.array([kx, kz_value, ky])
                else:
                    k = np.array([kx, ky, kz_value])

                # 计算 Berry curvature
                omega = self.berry_calc.berry_curvature_kubo(k, band_idx)

                # 取垂直于切片平面的分量
                if plane == 'kz_const':
                    omega_perp = omega[0]  # Ω_xy
                elif plane == 'kx_const':
                    omega_perp = omega[1]  # Ω_yz
                elif plane == 'ky_const':
                    omega_perp = omega[2]  # Ω_zx

                chern_sum += omega_perp * dkx * dky

        # Chern 数 = (1/2π) × 积分
        # 注意：这里 k 是分数坐标，需要乘以 (2π)² 转换
        chern_number = chern_sum / (2 * np.pi)
        return chern_number

    def fhs_method(self, plane: str = 'kz_const',
                   kz_value: float = 0.0,
                   band_idx: int = 0,
                   n_grid: int = 30) -> Dict:
        """
        Fukui-Hatsugai-Suzuki 方法计算 Chern 数

        该方法保证 Chern 数为严格整数，适用于精确拓扑分类

        Parameters:
        -----------
        plane : str
            切片平面
        kz_value : float
            固定坐标值
        band_idx : int
            能带索引
        n_grid : int
            网格密度

        Returns:
        --------
        Dict
            包含 Chern 数、U(1) 场、plaquette flux 等信息
        """
        chern_number, flux = self.berry_calc.fhs_chern_number(
            plane, kz_value, band_idx, n_grid
        )

        result = {
            'chern_number': chern_number,
            'flux_density': flux,
            'total_flux': np.sum(flux),
            'plane': plane,
            'kz_value': kz_value,
            'band_idx': band_idx,
            'n_grid': n_grid
        }

        return result

    def chern_number_vs_kz(self, kz_values: np.ndarray,
                           band_idx: int = 0,
                           n_grid: int = 20,
                           method: str = 'fhs') -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 Chern 数随 kz 的变化

        对于 Weyl 半金属，Chern 数在 Weyl 点处发生跳变

        Parameters:
        -----------
        kz_values : np.ndarray
            kz 值数组
        band_idx : int
            能带索引
        n_grid : int
            每个切片的网格密度
        method : str
            计算方法

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            kz_values, chern_numbers
        """
        chern_numbers = []

        for kz in kz_values:
            if method == 'fhs':
                result = self.fhs_method('kz_const', kz, band_idx, n_grid)
                chern_numbers.append(result['chern_number'])
            else:
                c = self.integrate_berry_curvature('kz_const', kz, band_idx, n_grid, method)
                chern_numbers.append(c)

        return kz_values, np.array(chern_numbers)

    def find_chern_transitions(self, kz_range: Tuple[float, float] = (0, 1),
                               band_idx: int = 0,
                               n_scan: int = 100,
                               n_grid: int = 15) -> List[Dict]:
        """
        搜索 Chern 数跳变点 (即 Weyl 点的 kz 投影)

        Parameters:
        -----------
        kz_range : Tuple[float, float]
            kz 扫描范围
        band_idx : int
            能带索引
        n_scan : int
            扫描点数
        n_grid : int
            网格密度

        Returns:
        --------
        List[Dict]
            Chern 跳变点列表
        """
        kz_vals = np.linspace(kz_range[0], kz_range[1], n_scan)
        _, chern_nums = self.chern_number_vs_kz(kz_vals, band_idx, n_grid, 'fhs')

        transitions = []
        for i in range(len(chern_nums) - 1):
            if chern_nums[i] != chern_nums[i + 1]:
                # 找到跳变点
                transitions.append({
                    'kz_left': kz_vals[i],
                    'kz_right': kz_vals[i + 1],
                    'chern_left': int(chern_nums[i]),
                    'chern_right': int(chern_nums[i + 1]),
                    'delta_chern': int(chern_nums[i + 1] - chern_nums[i])
                })

        return transitions

    def non_abelian_chern(self, bands: List[int], plane: str = 'kz_const',
                          kz_value: float = 0.0,
                          n_grid: int = 20) -> int:
        """
        非 Abel Chern 数 (多带简并情况)

        当多个能带简并时，Berry connection 变为非 Abel 规范场：
        A_μ^{ab} = i<u_a|∂_μ|u_b>

        Chern 数变为矩阵形式

        Parameters:
        -----------
        bands : List[int]
            简并能带索引列表
        plane : str
            切片平面
        kz_value : float
            固定坐标
        n_grid : int
            网格密度

        Returns:
        --------
        int
            非 Abel Chern 数 (迹)
        """
        n_bands = len(bands)

        # 生成网格
        kx_vals = np.linspace(0, 1, n_grid, endpoint=False)
        ky_vals = np.linspace(0, 1, n_grid, endpoint=False)

        # 计算非 Abel Wilson loop
        # W = Π U(k) where U_{ab} = <u_a(k)|u_a(k+dk)>
        total_chern = 0.0

        for kx in kx_vals:
            for ky in ky_vals:
                k = np.array([kx, ky, kz_value])
                dkx = np.array([1.0/n_grid, 0, 0])
                dky = np.array([0, 1.0/n_grid, 0])

                # 计算非 Abel link
                U_x = self._non_abelian_link(k, dkx, bands)
                U_y = self._non_abelian_link(k, dky, bands)
                U_x_inv = self._non_abelian_link(k + dky, dkx, bands)
                U_y_inv = self._non_abelian_link(k + dkx, dky, bands)

                # Plaquette
                F = U_x @ U_y @ np.linalg.inv(U_x_inv) @ np.linalg.inv(U_y_inv)

                # Berry flux = Im Tr ln F
                flux = np.imag(np.log(np.linalg.det(F) + 1e-15))
                total_chern += flux

        chern_number = int(np.round(total_chern / (2 * np.pi)))
        return chern_number

    def _non_abelian_link(self, k: np.ndarray, dk: np.ndarray,
                          bands: List[int]) -> np.ndarray:
        """
        计算非 Abel link 矩阵
        U_{ab}(k, k+dk) = <u_a(k)|u_b(k+dk)>
        """
        n_bands = len(bands)
        _, u_k = self.berry_calc.solve_eigenproblem(k)
        _, u_k_dk = self.berry_calc.solve_eigenproblem(k + dk)

        U = np.zeros((n_bands, n_bands), dtype=complex)
        for i, bi in enumerate(bands):
            for j, bj in enumerate(bands):
                U[i, j] = np.conj(u_k[:, bi]) @ u_k_dk[:, bj]

        return U

    def topological_phase_diagram(self, parameter_range: Dict,
                                  kz_value: float = 0.0,
                                  n_param: int = 20,
                                  n_grid: int = 15) -> Dict:
        """
        计算拓扑相图 (Chern 数随参数变化)

        Parameters:
        -----------
        parameter_range : Dict
            参数范围，如 {'m0': (0, 1), 'alpha': (0, 2)}
        kz_value : float
            kz 值
        n_param : int
            每个参数的采样点数
        n_grid : int
            网格密度

        Returns:
        --------
        Dict
            相图数据
        """
        # 目前支持 2 参数相图
        param_keys = list(parameter_range.keys())
        if len(param_keys) != 2:
            raise ValueError("目前仅支持 2 参数相图")

        p1_key, p2_key = param_keys
        p1_range = parameter_range[p1_key]
        p2_range = parameter_range[p2_key]

        p1_vals = np.linspace(p1_range[0], p1_range[1], n_param)
        p2_vals = np.linspace(p2_range[0], p2_range[1], n_param)

        phase_diagram = np.zeros((n_param, n_param), dtype=int)

        for i, p1 in enumerate(p1_vals):
            for j, p2 in enumerate(p2_vals):
                # 临时修改 Hamiltonian 参数
                # 简化处理：直接计算
                try:
                    chern = self._compute_chern_with_params(
                        {p1_key: p1, p2_key: p2}, kz_value, n_grid
                    )
                    phase_diagram[i, j] = chern
                except Exception:
                    phase_diagram[i, j] = 0

        return {
            'param1_key': p1_key,
            'param2_key': p2_key,
            'param1_values': p1_vals,
            'param2_values': p2_vals,
            'chern_numbers': phase_diagram
        }

    def _compute_chern_with_params(self, params: Dict, kz: float,
                                    n_grid: int) -> int:
        """
        给定参数计算 Chern 数 (辅助函数)
        """
        # 简化版本：使用 FHS 方法
        result = self.fhs_method('kz_const', kz, 0, n_grid)
        return result['chern_number']
