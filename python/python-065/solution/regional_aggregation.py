"""
regional_aggregation.py

基于 118_brc_naive 核心算法的区域气候统计聚合模块。

原项目 brc_naive 实现了十亿记录挑战的朴素解法，按城市聚合温度统计
（最小值、平均值、最大值）。

在本气候归因框架中，我们将其改造为按气候区域聚合极端事件统计量：
- 区域标识与匹配
- 区域平均异常强度
- 区域最大异常强度
- 区域事件频率
- 区域总面积/体积积分

核心公式：
- 区域平均：μ_R = (1/N_R) Σ_{i∈R} x_i
- 区域方差：σ²_R = (1/(N_R-1)) Σ_{i∈R} (x_i - μ_R)^2
- 区域极值：x_max,R = max_{i∈R} x_i
- 区域事件频率：f_R = N_extreme,R / N_R
- 区域总能量：E_R = Σ_{i∈R} |x_i| * A_i
"""

import numpy as np


def region_match(region_id, region_list, region_count):
    """
    在区域列表中查找给定区域的索引（基于 118_city_match）。

    Parameters
    ----------
    region_id : hashable
        待查找的区域标识。
    region_list : list
        已有区域列表。
    region_count : int
        当前区域数量。

    Returns
    -------
    int
        区域索引，若不存在则返回 -1。
    """
    for i in range(region_count):
        if region_list[i] == region_id:
            return i
    return -1


def aggregate_regional_statistics(region_ids, values, areas=None):
    """
    按区域聚合气候统计量。

    Parameters
    ----------
    region_ids : ndarray, shape (N,)
        每个格点的区域标识。
    values : ndarray, shape (N,)
        每个格点的气候异常值。
    areas : ndarray, shape (N,), optional
        每个格点的面积权重。

    Returns
    -------
    stats : dict
        {region_id: {"mean", "min", "max", "count", "sum", "var", "area_sum"}}
    """
    region_ids = np.asarray(region_ids)
    values = np.asarray(values)
    if areas is None:
        areas = np.ones_like(values)
    else:
        areas = np.asarray(areas)

    unique_ids = np.unique(region_ids)
    stats = {}

    for rid in unique_ids:
        mask = region_ids == rid
        vals = values[mask]
        w = areas[mask]

        count = len(vals)
        val_sum = np.sum(vals * w)
        area_sum = np.sum(w)
        mean_val = val_sum / area_sum if area_sum > 0 else 0.0
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))

        if count > 1:
            var_val = np.sum(w * (vals - mean_val) ** 2) / (np.sum(w) - np.mean(w))
        else:
            var_val = 0.0

        stats[int(rid)] = {
            "mean": float(mean_val),
            "min": min_val,
            "max": max_val,
            "count": count,
            "sum": float(val_sum),
            "var": float(var_val),
            "area_sum": float(area_sum),
        }

    return stats


def compute_regional_extreme_index(stats, weight_mean=0.3, weight_max=0.4,
                                    weight_freq=0.3):
    """
    计算综合区域极端事件指数（REI）。

    公式：
        REI = w_1 * (μ_R / μ_global) + w_2 * (x_max / x_max_global)
              + w_3 * f_R
    """
    global_mean = np.mean([s["mean"] for s in stats.values()])
    global_max = np.max([s["max"] for s in stats.values()]) if stats else 1.0

    if abs(global_mean) < 1e-14:
        global_mean = 1.0
    if abs(global_max) < 1e-14:
        global_max = 1.0

    rei = {}
    for rid, s in stats.items():
        rei[rid] = (
            weight_mean * (s["mean"] / global_mean)
            + weight_max * (s["max"] / global_max)
            + weight_freq * min(s["count"] / 100.0, 1.0)
        )
    return rei


def climate_region_summary(grid_regions, grid_anomalies, grid_areas=None):
    """
    生成完整的区域气候摘要报告。
    """
    stats = aggregate_regional_statistics(grid_regions, grid_anomalies, grid_areas)
    rei = compute_regional_extreme_index(stats)
    return {"stats": stats, "rei": rei}


def test_regional():
    regions = np.array([1, 1, 2, 2, 2, 3])
    values = np.array([10.0, 12.0, 5.0, 6.0, 7.0, 20.0])
    stats = aggregate_regional_statistics(regions, values)
    assert stats[1]["mean"] == 11.0
    assert stats[2]["mean"] == 6.0
    assert stats[3]["max"] == 20.0
    print("regional_aggregation 自测试通过")


if __name__ == "__main__":
    test_regional()
