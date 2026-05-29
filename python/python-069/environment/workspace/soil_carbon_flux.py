"""
土壤-冠层界面碳通量模块：基于 quadrilateral_witherden_rule，
在土壤-冠层界面的四边形区域上进行高阶数值求积，
计算生态系统净碳交换量（NEE）。

核心公式：
  净生态系统交换量（NEE）：
      NEE = F_c + F_s - GPP
  其中：
      F_c: 冠层呼吸通量
      F_s: 土壤呼吸通量
      GPP: 总初级生产力

  土壤呼吸（Lloyd-Taylor 模型）：
      R_s = R_10 * exp( E_0 * (1/(T_ref - T_0) - 1/(T_soil - T_0)) )
      T_ref = 283.15 K, T_0 = 227.13 K

  冠层呼吸：
      R_c = integral_{Omega} R_d(x,y) * LAI(x,y) dOmega

  使用 Witherden 高阶求积在 [-1,1]^2 映射后的四边形上积分。
"""
import numpy as np


def quadrilateral_witherden_rule(p):
    """
    返回四边形 [-1,1]^2 上的 Witherden 型高阶求积规则。
    返回: n, x, y, w
    """
    p = int(p)
    if p <= 1:
        n = 1
        x = np.array([0.0])
        y = np.array([0.0])
        w = np.array([4.0])
    elif p <= 3:
        n = 4
        a = 1.0 / np.sqrt(3.0)
        x = np.array([-a, a, -a, a])
        y = np.array([-a, -a, a, a])
        w = np.array([1.0, 1.0, 1.0, 1.0])
    elif p <= 5:
        n = 9
        a = np.sqrt(3.0 / 5.0)
        x = np.array([-a, 0.0, a, -a, 0.0, a, -a, 0.0, a])
        y = np.array([-a, -a, -a, 0.0, 0.0, 0.0, a, a, a])
        wg = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        w = np.array([wg[i] * wg[j] for i in range(3) for j in range(3)])
        w = np.array(w) * (25.0 / 9.0) / np.sum(w) * 4.0  # 归一化到总面积 4
    else:
        # 更高阶：7x7 Gauss-Legendre 乘积规则
        n1d = 7
        xi, wi = np.polynomial.legendre.leggauss(n1d)
        n = n1d * n1d
        x = np.zeros(n)
        y = np.zeros(n)
        w = np.zeros(n)
        idx = 0
        for i in range(n1d):
            for j in range(n1d):
                x[idx] = xi[i]
                y[idx] = xi[j]
                w[idx] = wi[i] * wi[j]
                idx += 1
    return n, x, y, w


def map_quad_to_physical(xi, eta, corners):
    """
    将参考四边形 [-1,1]^2 映射到物理四边形。
    corners: (4,2) 四个角点坐标
    """
    N1 = (1.0 - xi) * (1.0 - eta) / 4.0
    N2 = (1.0 + xi) * (1.0 - eta) / 4.0
    N3 = (1.0 + xi) * (1.0 + eta) / 4.0
    N4 = (1.0 - xi) * (1.0 + eta) / 4.0
    x = N1 * corners[0, 0] + N2 * corners[1, 0] + N3 * corners[2, 0] + N4 * corners[3, 0]
    y = N1 * corners[0, 1] + N2 * corners[1, 1] + N3 * corners[2, 1] + N4 * corners[3, 1]
    return x, y


def jacobian_quad(xi, eta, corners):
    """计算等参映射的雅可比行列式。"""
    dN_dxi = np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]) / 4.0
    dN_deta = np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]) / 4.0
    dx_dxi = np.sum(dN_dxi * corners[:, 0])
    dx_deta = np.sum(dN_deta * corners[:, 0])
    dy_dxi = np.sum(dN_dxi * corners[:, 1])
    dy_deta = np.sum(dN_deta * corners[:, 1])
    jac = abs(dx_dxi * dy_deta - dx_deta * dy_dxi)
    return max(jac, 1e-14)


def integrate_canopy_respiration(corners, n, xq, yq, wq,
                                 lai_func, rd_func):
    """
    在四边形区域上积分冠层呼吸通量。
    """
    total = 0.0
    for i in range(n):
        x, y = map_quad_to_physical(xq[i], yq[i], corners)
        jac = jacobian_quad(xq[i], yq[i], corners)
        lai = lai_func(x, y)
        rd = rd_func(x, y)
        total += wq[i] * jac * rd * lai
    return total


def lloyd_taylor_soil_respiration(t_soil_c, r10=2.0, e0=308.56):
    """
    Lloyd-Taylor 土壤呼吸模型 (umol/m^2/s)。
    t_soil_c: 土壤温度 (°C)
    """
    # TODO: 实现 Lloyd-Taylor 土壤呼吸模型
    # 关键公式：
    #   R_s = R_10 * exp( E_0 * (1/(T_ref - T_0) - 1/(T_soil - T_0)) )
    #   T_ref = 283.15 K, T_0 = 227.13 K
    #   T_soil 需从 °C 转换为 K
    raise NotImplementedError("Hole 2: 请补全 Lloyd-Taylor 土壤呼吸公式")


def compute_nee(gpp, canopy_resp, soil_resp):
    """
    计算净生态系统交换量 NEE。
    NEE > 0: 碳排放
    NEE < 0: 碳吸收
    """
    return canopy_resp + soil_resp - gpp
