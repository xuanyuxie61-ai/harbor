
import numpy as np


def region_match(region_id, region_list, region_count):
    for i in range(region_count):
        if region_list[i] == region_id:
            return i
    return -1


def aggregate_regional_statistics(region_ids, values, areas=None):
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
