"""
reion_scenarios.py
==================
多场景再电离模拟驱动

本模块运行多个不同物理假设的再电离场景, 并比较结果. 场景差异主要体现在:
  - 源类型 (Pop II 星, Pop III 星, QSO)
  - UVB 振幅
  - 逃逸分数
  - Clumping factor 模型

物理背景:
  再电离的源性质存在很大不确定性. Pop II 星 (普通大质量星) 被认为
  是主要贡献者, 但 Pop III 星 (极低金属丰度, 大质量) 和 QSO (活动星系核)
  在某些模型中也扮演重要角色. 不同源类型具有不同的谱形状与逃逸分数.

对应种子项目:
  - 1155_anhkiet120206-lgtm_Supplementary-Materials-for-EEIO-Analysis
    (多场景 IO 分析 → 多场景再电离参数研究)
"""

import numpy as np
from reion_solver import run_single_simulation
from reion_cosmology import tau_e_obs, sigma_tau


# =============================================================================
# 场景定义
# =============================================================================
SCENARIOS = {
    "pop2_standard": {
        "description": "标准 Pop II 星主导模型",
        "uvb_amplitude": 1.5e-15,
        "f_esc": 0.15,
        "alpha_spec": 1.5,
        "epsilon_star": 0.1,
        "clumping_model": "constant",
        "clumping_value": 3.0,
    },
    "pop3_efficient": {
        "description": "Pop III 星高效电离模型",
        "uvb_amplitude": 6.0e-15,
        "f_esc": 0.30,
        "alpha_spec": 2.5,
        "epsilon_star": 0.05,
        "clumping_model": "constant",
        "clumping_value": 5.0,
    },
    "qso_dominated": {
        "description": "QSO 主导模型",
        "uvb_amplitude": 3.0e-15,
        "f_esc": 0.70,
        "alpha_spec": 1.8,
        "epsilon_star": 0.01,
        "clumping_model": "power_law",
        "clumping_value": 4.0,
    },
    "late_reionization": {
        "description": "晚再电离模型 (z_1/2 ~ 7)",
        "uvb_amplitude": 8.0e-16,
        "f_esc": 0.05,
        "alpha_spec": 1.5,
        "epsilon_star": 0.08,
        "clumping_model": "constant",
        "clumping_value": 2.0,
    },
    "early_reionization": {
        "description": "早再电离模型 (z_1/2 ~ 9)",
        "uvb_amplitude": 5.0e-15,
        "f_esc": 0.15,
        "alpha_spec": 2.0,
        "epsilon_star": 0.15,
        "clumping_model": "constant",
        "clumping_value": 4.0,
    },
    "high_clumping": {
        "description": "高 clumping 模型 (C=10)",
        "uvb_amplitude": 2.5e-15,
        "f_esc": 0.12,
        "alpha_spec": 1.5,
        "epsilon_star": 0.1,
        "clumping_model": "constant",
        "clumping_value": 10.0,
    },
}


def run_scenario(scenario_name, N_grid=64, N_steps=80, verbose=True):
    """运行单个场景的模拟.

    Parameters
    ----------
    scenario_name : str
        场景名 (需在 SCENARIOS 中)
    N_grid, N_steps : int
    verbose : bool

    Returns
    -------
    result : dict
    summary : dict
    """
    if scenario_name not in SCENARIOS:
        raise ValueError("未知场景: %s, 可选: %s" % (
            scenario_name, list(SCENARIOS.keys())))

    params = SCENARIOS[scenario_name].copy()
    desc = params.pop("description")
    if verbose:
        print("\n>>> 场景: %s" % scenario_name)
        print("    %s" % desc)

    result = run_single_simulation(
        scenario_name=scenario_name,
        N_grid=N_grid,
        N_steps=N_steps,
        **params,
    )

    summary = {
        "z_final": float(result["z_array"][-1]),
        "mean_xHII_final": float(result["mean_xHII"][-1]),
        "mean_tau_eff": float(result["mean_tau_eff"]),
        "z_half_reionization": float(result["z_half_reionization"]),
        "tau_obs_planck": tau_e_obs,
        "scenario_description": desc,
    }
    return result, summary


def run_all_scenarios(N_grid=64, N_steps=80):
    """运行所有预定义场景.

    Returns
    -------
    all_results : dict
    all_summaries : dict
    comparison : dict
    """
    all_results = {}
    all_summaries = {}

    print("\n" + "=" * 70)
    print("多场景再电离模拟")
    print("共 %d 个场景" % len(SCENARIOS))
    print("=" * 70)

    for name in SCENARIOS:
        try:
            result, summary = run_scenario(name, N_grid=N_grid, N_steps=N_steps)
            all_results[name] = result
            all_summaries[name] = summary
        except Exception as e:
            print("  场景 %s 失败: %s" % (name, str(e)))
            all_results[name] = None
            all_summaries[name] = {"error": str(e)}

    # 比较分析
    comparison = compare_scenarios(all_summaries)

    # 打印汇总
    print("\n" + "=" * 70)
    print("场景比较汇总:")
    print("-" * 70)
    print("%-20s %8s %10s %10s %10s" % (
        "场景", "z_final", "<xHII>", "tau_eff", "z_{1/2}"))
    print("-" * 70)
    for name, s in all_summaries.items():
        if "error" not in s:
            print("%-20s %8.2f %10.4f %10.4f %10.2f" % (
                name, s["z_final"], s["mean_xHII_final"],
                s["mean_tau_eff"], s["z_half_reionization"]))
    print("=" * 70)

    return all_results, all_summaries, comparison


def compare_scenarios(summaries):
    """比较多个场景的结果.

    Parameters
    ----------
    summaries : dict

    Returns
    -------
    comparison : dict
    """
    valid = {k: v for k, v in summaries.items() if "error" not in v}
    if not valid:
        return {"error": "无有效场景"}

    names = list(valid.keys())
    tau_values = np.array([v["mean_tau_eff"] for v in valid.values()])
    z_half_values = np.array([v["z_half_reionization"] for v in valid.values()])

    # 与 Planck tau 的 chi^2
    chi2_values = ((tau_values - tau_e_obs) / max(sigma_tau, 1.0e-10)) ** 2
    best_idx = int(np.argmin(chi2_values))

    return {
        "best_fit_scenario": names[best_idx],
        "best_chi2": float(chi2_values[best_idx]),
        "tau_range": (float(np.min(tau_values)), float(np.max(tau_values))),
        "z_half_range": (float(np.min(z_half_values)),
                         float(np.max(z_half_values))),
        "tau_spread": float(np.std(tau_values)),
        "n_scenarios": len(valid),
    }


def analyze_parameter_sensitivity(param_name, values,
                                  base_scenario="pop2_standard",
                                  N_grid=32, N_steps=50):
    """分析单个参数的灵敏度.

    固定其他参数, 扫描 param_name, 记录 xHII_mean 与 tau_eff.

    Parameters
    ----------
    param_name : str
    values : array
    base_scenario : str
    N_grid, N_steps : int

    Returns
    -------
    sensitivity : dict
    """
    base_params = SCENARIOS[base_scenario].copy()
    base_params.pop("description", None)

    results = []
    for v in values:
        params = base_params.copy()
        params[param_name] = float(v)
        try:
            result = run_single_simulation(
                scenario_name="%s_sweep" % param_name,
                N_grid=N_grid,
                N_steps=N_steps,
                **params,
            )
            results.append({
                "value": float(v),
                "mean_xHII": float(result["mean_xHII"][-1]),
                "tau_eff": float(result["mean_tau_eff"]),
                "z_half": float(result["z_half_reionization"]),
            })
        except Exception:
            results.append({
                "value": float(v),
                "mean_xHII": np.nan,
                "tau_eff": np.nan,
                "z_half": np.nan,
            })
    return {"param_name": param_name, "sweep_results": results}


def get_scenario_metadata(name):
    """获取场景元数据."""
    if name in SCENARIOS:
        return SCENARIOS[name].copy()
    return None


def list_scenarios():
    """列出所有可用场景."""
    return list(SCENARIOS.keys())
