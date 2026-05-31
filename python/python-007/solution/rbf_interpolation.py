"""
径向基函数插值模块
整合自：1014_rbf_interp_2d（RBF二维插值）

在吸积盘模拟中用于：
  1. 在非结构网格上重构密度、压力和速度场
  2. 从稀疏采样点插值到精细网格
  3. 喷流边界附近的场平滑处理
"""
import numpy as np


def phi_multiquadric(r, r0):
    """
    多二次径向基函数（Multiquadric）：
        phi(r) = sqrt(r^2 + r0^2)

    对于薄板样条和高斯基函数，Multiquadric 通常具有最好的插值精度。
    """
    return np.sqrt(r * r + r0 * r0)


def phi_inverse_multiquadric(r, r0):
    """
    逆多二次径向基函数：
        phi(r) = 1 / sqrt(r^2 + r0^2)
    """
    return 1.0 / np.sqrt(r * r + r0 * r0)


def phi_thin_plate_spline(r, r0):
    """
    薄板样条径向基函数：
        phi(r) = r^2 * log(r / r0)   (r != 0)
        phi(0) = 0
    """
    result = np.zeros_like(r)
    mask = r > 1e-15
    result[mask] = r[mask] ** 2 * np.log(r[mask] / r0)
    return result


def phi_gaussian(r, r0):
    """
    高斯径向基函数：
        phi(r) = exp(-0.5 * r^2 / r0^2)
    """
    return np.exp(-0.5 * r * r / (r0 * r0))


def rbf_weights(data_points, data_values, r0=1.0, basis='multiquadric'):
    """
    计算 RBF 插值权重 w。

    给定 N_d 个散乱数据点 x_d 和函数值 f_d，构造插值：
        f(x) = sum_{j=1}^{N_d} w_j * phi(||x - x_d_j||)

    通过配点法求解线性系统：
        A * w = f_d
    其中 A_{ij} = phi(||x_d_i - x_d_j||)。

    参数:
        data_points: (N_d, M) 数据点坐标
        data_values: (N_d,) 函数值
        r0: 形状参数
        basis: 'multiquadric', 'inverse_mq', 'tps', 'gaussian'

    返回:
        weights: (N_d,) 权重向量
    """
    data_points = np.asarray(data_points, dtype=np.float64)
    data_values = np.asarray(data_values, dtype=np.float64)
    nd = data_points.shape[0]

    if len(data_values) != nd:
        raise ValueError("data_values length must match data_points rows")

    # 选择基函数
    basis_funcs = {
        'multiquadric': phi_multiquadric,
        'inverse_mq': phi_inverse_multiquadric,
        'tps': phi_thin_plate_spline,
        'gaussian': phi_gaussian
    }
    if basis not in basis_funcs:
        raise ValueError(f"Unknown basis: {basis}")
    phi_func = basis_funcs[basis]

    # 构建配点矩阵
    A = np.zeros((nd, nd), dtype=np.float64)
    for i in range(nd):
        diff = data_points[i] - data_points
        r = np.linalg.norm(diff, axis=1)
        A[i, :] = phi_func(r, r0)

    # 数值稳定性：添加小正则化
    reg = 1e-10 * np.eye(nd)
    A += reg

    # 求解
    weights = np.linalg.solve(A, data_values)
    return weights


def rbf_interpolate(query_points, data_points, weights, r0=1.0, basis='multiquadric'):
    """
    使用已计算的权重在查询点处求 RBF 插值。

    参数:
        query_points: (N_q, M) 查询点
        data_points: (N_d, M) 原始数据点
        weights: (N_d,) RBF 权重
        r0: 形状参数
        basis: 基函数类型

    返回:
        values: (N_q,) 插值结果
    """
    query_points = np.asarray(query_points, dtype=np.float64)
    data_points = np.asarray(data_points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    nq = query_points.shape[0]
    nd = data_points.shape[0]

    basis_funcs = {
        'multiquadric': phi_multiquadric,
        'inverse_mq': phi_inverse_multiquadric,
        'tps': phi_thin_plate_spline,
        'gaussian': phi_gaussian
    }
    phi_func = basis_funcs[basis]

    values = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        diff = query_points[i] - data_points
        r = np.linalg.norm(diff, axis=1)
        phi_vals = phi_func(r, r0)
        values[i] = np.dot(weights, phi_vals)

    return values


def rbf_gradient_2d(query_point, data_points, weights, r0=1.0, basis='multiquadric', h=1e-6):
    """
    使用数值微分计算 RBF 场在2D查询点处的梯度。

    梯度公式（数值近似）：
        df/dx ≈ [f(x+h,y) - f(x-h,y)] / (2h)
        df/dy ≈ [f(x,y+h) - f(x,y-h)] / (2h)
    """
    qp = np.asarray(query_point, dtype=np.float64).reshape(1, -1)

    # x方向
    qp_xp = qp.copy()
    qp_xp[0, 0] += h
    qp_xm = qp.copy()
    qp_xm[0, 0] -= h

    # y方向
    qp_yp = qp.copy()
    qp_yp[0, 1] += h
    qp_ym = qp.copy()
    qp_ym[0, 1] -= h

    fx = (rbf_interpolate(qp_xp, data_points, weights, r0, basis)[0] -
          rbf_interpolate(qp_xm, data_points, weights, r0, basis)[0]) / (2 * h)
    fy = (rbf_interpolate(qp_yp, data_points, weights, r0, basis)[0] -
          rbf_interpolate(qp_ym, data_points, weights, r0, basis)[0]) / (2 * h)

    return np.array([fx, fy])
