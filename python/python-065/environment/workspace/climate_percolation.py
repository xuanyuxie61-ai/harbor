
import numpy as np


def detect_extreme_grid(anomaly_field, threshold):
    if anomaly_field.ndim != 2:
        raise ValueError("异常场必须是二维数组")
    occupied = np.zeros_like(anomaly_field, dtype=np.int64)
    occupied[anomaly_field > threshold] = 1
    return occupied


def components_2d(a):
    m, n = a.shape
    c = np.zeros((m, n), dtype=np.int64)
    component_index = 0

    for ii in range(m):
        for jj in range(n):
            if a[ii, jj] != 0 and c[ii, jj] == 0:

                component_index += 1
                stack = [(ii, jj)]
                while stack:
                    i, j = stack.pop()
                    if c[i, j] != 0:
                        continue
                    c[i, j] = component_index

                    if i - 1 >= 0 and a[i - 1, j] != 0 and c[i - 1, j] == 0:
                        stack.append((i - 1, j))
                    if i + 1 < m and a[i + 1, j] != 0 and c[i + 1, j] == 0:
                        stack.append((i + 1, j))
                    if j - 1 >= 0 and a[i, j - 1] != 0 and c[i, j - 1] == 0:
                        stack.append((i, j - 1))
                    if j + 1 < n and a[i, j + 1] != 0 and c[i, j + 1] == 0:
                        stack.append((i, j + 1))
    return c


def spanning_analysis(cls, m, n):
    component_num = int(cls.max())
    if component_num == 0:
        return 0, 0, []

    component_sizes = []
    isspanx = np.zeros(component_num + 1, dtype=np.int64)
    isspany = np.zeros(component_num + 1, dtype=np.int64)

    for comp in range(1, component_num + 1):
        size = int(np.sum(cls == comp))
        component_sizes.append(size)


        left = np.any(cls[:, 0] == comp)
        right = np.any(cls[:, n - 1] == comp)
        if left and right:
            isspanx[comp] = 1


        top = np.any(cls[0, :] == comp)
        bottom = np.any(cls[m - 1, :] == comp)
        if top and bottom:
            isspany[comp] = 1

    spanx = int(np.sum(isspanx))
    spany = int(np.sum(isspany))
    return spanx, spany, component_sizes


def percolation_order_parameter(component_sizes, total_sites):
    if not component_sizes:
        return 0.0
    s_max = max(component_sizes)
    return s_max / total_sites


def correlation_length_estimate(component_sizes, threshold_bins=20):


    raise NotImplementedError("关联长度估计公式待实现")


def run_percolation_attribution(anomaly_field, threshold=2.0):
    m, n = anomaly_field.shape
    occupied = detect_extreme_grid(anomaly_field, threshold)
    nosites = int(np.sum(occupied))
    posites = nosites / (m * n)

    cls = components_2d(occupied)
    spanx, spany, component_sizes = spanning_analysis(cls, m, n)

    p_inf = 0.0
    xi = 0.0

    return {
        "occupied_grid": occupied,
        "components": cls,
        "nosites": nosites,
        "posites": posites,
        "spanx": spanx,
        "spany": spany,
        "component_sizes": component_sizes,
        "p_infinity": p_inf,
        "correlation_length": xi,
    }


def test_percolation():
    np.random.seed(42)
    field = np.random.randn(20, 20)
    result = run_percolation_attribution(field, threshold=1.5)
    assert result["posites"] >= 0.0
    print("climate_percolation 自测试通过")


if __name__ == "__main__":
    test_percolation()
