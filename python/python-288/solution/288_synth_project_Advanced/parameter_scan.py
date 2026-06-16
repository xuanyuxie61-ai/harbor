"""
parameter_scan.py - 参数扫描与不确定性分析

本模块融合以下种子项目的核心算法：
  - 832_ode_sweep_parfor → 参数扫描框架
  - 561_hypercube_surface_distance → 高维采样与统计

功能：
  1. 等离子体参数空间扫描
  2. 偏滤器热负荷参数依赖性
  3. Monte Carlo不确定性传播
  4. 高维参数空间采样
  5. 统计分析与敏感性指标
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from plasma_physics import (
    two_point_model, divertor_target_heat_flux,
    sound_speed, bohm_diffusion, coulomb_logarithm,
    spitzer_harm_conductivity, EV_TO_JOULE
)


# =============================================================================
# 参数扫描引擎（来自832_ode_sweep_parfor）
# =============================================================================
class ParameterScanner:
    """
    参数空间扫描引擎

    用于系统性地研究偏滤器热负荷对不同等离子体参数的依赖关系
    """

    def __init__(self):
        self.results = []
        self.parameter_names = []
        self.parameter_ranges = []

    def add_parameter(self, name: str, values: np.ndarray):
        """添加扫描参数"""
        self.parameter_names.append(name)
        self.parameter_ranges.append(values)

    def scan(self, func: Callable, verbose: bool = False) -> np.ndarray:
        """
        执行全因子参数扫描

        对于N个参数，每个有M个值，总共需要M^N次计算

        参数:
            func: 计算函数，接受参数字典返回结果
            verbose: 是否打印进度
        返回:
            results: 所有参数组合的结果数组
        """
        # 生成全因子组合
        grids = np.meshgrid(*self.parameter_ranges, indexing='ij')
        shapes = [len(r) for r in self.parameter_ranges]
        n_total = int(np.prod(shapes))

        results = []
        flat_grids = [g.ravel() for g in grids]

        for idx in range(n_total):
            params = {}
            for p, name in enumerate(self.parameter_names):
                params[name] = flat_grids[p][idx]

            try:
                result = func(params)
                results.append(result)
            except Exception as e:
                results.append(None)
                if verbose:
                    print(f"  参数组合 {idx} 计算失败: {e}")

            if verbose and (idx + 1) % max(1, n_total // 10) == 0:
                print(f"  进度: {idx + 1}/{n_total}")

        self.results = results
        return np.array(results, dtype=object)


def scan_divertor_heat_flux(n_upstream_range: np.ndarray,
                              T_upstream_range: np.ndarray,
                              L_parallel: float = 10.0) -> Dict:
    """
    扫描偏滤器靶板热负荷 vs 上游密度和温度

    参数:
        n_upstream_range: 上游密度范围 [m^-3]
        T_upstream_range: 上游温度范围 [eV]
        L_parallel: 平行连接长度 [m]
    返回:
        dict包含扫描结果
    """
    n_n = len(n_upstream_range)
    n_T = len(T_upstream_range)

    q_target_grid = np.zeros((n_n, n_T))
    T_target_grid = np.zeros((n_n, n_T))

    for i, n_u in enumerate(n_upstream_range):
        for j, T_u in enumerate(T_upstream_range):
            # 估算平行热流
            kappa = spitzer_harm_conductivity(n_u, T_u)
            q_parallel = (2.0 / 7.0) * kappa * T_u**3.5 / L_parallel

            # 两点模型
            result = two_point_model(q_parallel, n_u, T_u, L_parallel)

            q_target_grid[i, j] = result['q_target_Wm2']
            T_target_grid[i, j] = result['T_target_eV']

    return {
        'n_upstream': n_upstream_range,
        'T_upstream': T_upstream_range,
        'q_target': q_target_grid,
        'T_target': T_target_grid,
    }


def scan_recycling_effects(recycling_range: np.ndarray,
                             n_upstream: float = 3e19,
                             T_upstream: float = 100.0,
                             L_parallel: float = 10.0) -> Dict:
    """
    扫描再循环系数对靶板条件的影响

    参数:
        recycling_range: 再循环系数范围 [0, 1]
        n_upstream: 上游密度
        T_upstream: 上游温度
        L_parallel: 连接长度
    返回:
        dict包含扫描结果
    """
    n_R = len(recycling_range)
    results = {
        'recycling': recycling_range,
        'q_target': np.zeros(n_R),
        'T_target': np.zeros(n_R),
        'n_target': np.zeros(n_R),
    }

    for i, R in enumerate(recycling_range):
        kappa = spitzer_harm_conductivity(n_upstream, T_upstream)
        q_parallel = (2.0 / 7.0) * kappa * T_upstream**3.5 / L_parallel

        result = two_point_model(q_parallel, n_upstream, T_upstream,
                                   L_parallel, recycling_R=R)

        results['q_target'][i] = result['q_target_Wm2']
        results['T_target'][i] = result['T_target_eV']
        results['n_target'][i] = result['n_target']

    return results


# =============================================================================
# Monte Carlo不确定性分析（来自561_hypercube_surface_distance的采样思想）
# =============================================================================
def monte_carlo_uncertainty(n_samples: int = 1000,
                              param_distributions: Optional[Dict] = None,
                              L_parallel: float = 10.0) -> Dict:
    """
    Monte Carlo不确定性传播分析

    对于每个输入参数赋予概率分布，通过随机采样
    计算输出（靶板热负荷）的概率分布

    采样策略借鉴超立方体表面距离计算中的
    高维均匀采样方法

    参数:
        n_samples: 采样数量
        param_distributions: 参数分布字典
        L_parallel: 连接长度
    返回:
        dict包含统计结果
    """
    if param_distributions is None:
        # 默认参数分布
        param_distributions = {
            'n_upstream': {'type': 'lognormal', 'mean': 3e19, 'std': 1e19},
            'T_upstream': {'type': 'normal', 'mean': 100.0, 'std': 20.0},
            'lambda_q': {'type': 'normal', 'mean': 0.003, 'std': 0.001},
            'gamma_sheath': {'type': 'normal', 'mean': 7.0, 'std': 0.5},
        }

    # 采样
    samples = {}
    for name, dist in param_distributions.items():
        if dist['type'] == 'normal':
            samples[name] = np.random.normal(dist['mean'], dist['std'], n_samples)
        elif dist['type'] == 'lognormal':
            # 从均值和标准差推导lognormal参数
            mu = np.log(dist['mean']**2 / np.sqrt(dist['std']**2 + dist['mean']**2 + 1e-30))
            sigma = np.sqrt(np.log(1 + dist['std']**2 / (dist['mean']**2 + 1e-30)))
            samples[name] = np.random.lognormal(mu, sigma, n_samples)
        elif dist['type'] == 'uniform':
            samples[name] = np.random.uniform(dist['low'], dist['high'], n_samples)
        else:
            samples[name] = np.full(n_samples, dist['mean'])

    # 物理约束：确保参数非负
    for name in samples:
        samples[name] = np.abs(samples[name])

    # 计算每个样本的热负荷
    q_targets = np.zeros(n_samples)
    T_targets = np.zeros(n_samples)

    for i in range(n_samples):
        n_u = samples.get('n_upstream', np.full(n_samples, 3e19))[i]
        T_u = samples.get('T_upstream', np.full(n_samples, 100.0))[i]

        # 确保物理合理
        n_u = max(n_u, 1e17)
        T_u = max(T_u, 1.0)

        kappa = spitzer_harm_conductivity(n_u, T_u)
        q_parallel = (2.0 / 7.0) * kappa * T_u**3.5 / L_parallel

        try:
            result = two_point_model(q_parallel, n_u, T_u, L_parallel)
            q_targets[i] = result['q_target_Wm2']
            T_targets[i] = result['T_target_eV']
        except Exception:
            q_targets[i] = 0.0
            T_targets[i] = 0.0

    # 统计分析
    results = {
        'q_target_mean': np.mean(q_targets),
        'q_target_std': np.std(q_targets),
        'q_target_median': np.median(q_targets),
        'q_target_5th': np.percentile(q_targets, 5),
        'q_target_95th': np.percentile(q_targets, 95),
        'T_target_mean': np.mean(T_targets),
        'T_target_std': np.std(T_targets),
        'n_samples': n_samples,
        'samples': samples,
        'q_targets': q_targets,
        'T_targets': T_targets,
    }

    return results


# =============================================================================
# 超立方体采样（来自561_hypercube_surface_distance）
# =============================================================================
def hypercube_surface_sample(n_dimensions: int, n_samples: int,
                                side_length: float = 1.0) -> np.ndarray:
    """
    在D维超立方体表面上均匀采样

    算法：
    1. 随机选择一个维度（面）
    2. 将该维度设为0或1
    3. 其他维度在[0,1]上均匀采样

    参数:
        n_dimensions: 维度数
        n_samples: 采样数
        side_length: 立方体边长
    返回:
        samples: shape (n_samples, n_dimensions)
    """
    samples = np.zeros((n_samples, n_dimensions))

    for i in range(n_samples):
        # 随机选择面
        face_dim = np.random.randint(0, n_dimensions)
        face_value = np.random.choice([0.0, side_length])

        # 该面维度固定
        sample = np.random.uniform(0, side_length, n_dimensions)
        sample[face_dim] = face_value
        samples[i] = sample

    return samples


def compute_pairwise_distance_stats(samples: np.ndarray) -> Dict:
    """
    计算采样点之间的距离统计

    参数:
        samples: shape (n_samples, n_dim)
    返回:
        dict包含均值、方差等
    """
    n = len(samples)
    distances = []

    # 随机抽样计算（避免O(n^2)）
    n_pairs = min(n * (n - 1) // 2, 10000)

    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        d = np.linalg.norm(samples[i] - samples[j])
        distances.append(d)

    distances = np.array(distances)

    return {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'median_distance': float(np.median(distances)),
    }


# =============================================================================
# 敏感性分析
# =============================================================================
def sensitivity_analysis(base_params: Dict,
                           param_to_vary: str,
                           variation_range: np.ndarray,
                           output_func: Callable) -> Dict:
    """
    单参数敏感性分析

    变化一个参数，保持其他参数固定，观察输出变化

    参数:
        base_params: 基准参数
        param_to_vary: 要变化的参数名
        variation_range: 变化范围
        output_func: 输出计算函数
    返回:
        dict包含敏感性指标
    """
    outputs = np.zeros(len(variation_range))

    for i, val in enumerate(variation_range):
        params = base_params.copy()
        params[param_to_vary] = val
        outputs[i] = output_func(params)

    # 敏感性 = 输出变化 / 输入变化
    input_range = variation_range[-1] - variation_range[0]
    output_range = np.max(outputs) - np.min(outputs)
    base_output = outputs[len(outputs) // 2]

    sensitivity = output_range / (input_range + 1e-30)
    relative_sensitivity = sensitivity * base_params.get(param_to_vary, 1.0) / (base_output + 1e-30)

    return {
        'variation_range': variation_range,
        'outputs': outputs,
        'sensitivity': float(sensitivity),
        'relative_sensitivity': float(relative_sensitivity),
        'output_range': float(output_range),
        'base_output': float(base_output),
    }
