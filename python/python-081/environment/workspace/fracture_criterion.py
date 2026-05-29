"""
fracture_criterion.py
博士级大变形非线性有限元分析 — 断裂准则与损伤阈值判定模块

融合原项目:
  - 586_image_threshold: 图像阈值分割

核心数学:
  将损伤场类比为灰度图像，通过阈值判定将连续损伤变量场
  转换为二元断裂状态场（断裂/未断裂）

  1. 损伤阈值判定:
     给定损伤场 D(x) ∈ [0, 1]，选择阈值 D_c:
       若 D(x) >= D_c:  单元断裂 (状态 = 1)
       若 D(x) <  D_c:  单元完整 (状态 = 0)

  2. Otsu 自动阈值法（可选）:
     最大化类间方差:
       σ_b^2(t) = ω_0(t) ω_1(t) [μ_0(t) - μ_1(t)]^2
     其中:
       ω_0 = 背景像素比例, μ_0 = 背景均值
       ω_1 = 前景像素比例, μ_1 = 前景均值

  3. 断裂能释放率准则:
     G = -∂Π/∂A = ∫_Ω Y dD / ∂A
     其中 Y = -∂Ψ/∂D 为损伤驱动力，A 为裂纹面积

  4. 裂纹路径提取:
     通过断裂单元连通性分析提取裂纹前沿
"""

import numpy as np


def threshold_damage_field(damage_values, threshold):
    """
    对损伤场进行阈值分割

    源自原项目 586_image_threshold (image_threshold)

    输入:
        damage_values: (N,) 单元损伤值数组
        threshold: 损伤阈值 D_c ∈ [0, 1]
    输出:
        fractured: (N,) bool 数组，True 表示断裂
    """
    damage_values = np.array(damage_values, dtype=float)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    fractured = damage_values >= threshold
    return fractured


def otsu_threshold(damage_values):
    """
    Otsu 自动阈值选择

    数学:
      对损伤值在 [0,1] 范围内均匀分 bin，寻找最优阈值 t
      最大化: σ_b^2(t) = ω_0 ω_1 (μ_0 - μ_1)^2
    """
    vals = np.array(damage_values, dtype=float)
    vals = np.clip(vals, 0.0, 1.0)

    n_bins = 256
    hist, bin_edges = np.histogram(vals, bins=n_bins, range=(0.0, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total = len(vals)
    total_sum = np.sum(vals)

    max_var = -1.0
    best_t = 0.5

    w0 = 0
    sum0 = 0.0

    for i in range(n_bins):
        w0 += hist[i]
        sum0 += hist[i] * bin_centers[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        mu0 = sum0 / w0
        mu1 = (total_sum - sum0) / w1
        var_between = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        if var_between > max_var:
            max_var = var_between
            best_t = bin_centers[i]

    return float(best_t)


def fracture_energy_release_rate(damage_history, strain_energy_history, element_volume):
    """
    计算断裂能释放率 G

    数学:
      G ≈ ΔΨ_dissipated / ΔA
      ΔΨ_dissipated = Σ Y_i ΔD_i V_i
    """
    dD = np.diff(damage_history, prepend=0.0)
    # 简化的损伤驱动力: Y ≈ Ψ_e / (1 - D)
    Y = np.zeros_like(damage_history)
    for i in range(len(damage_history)):
        denom = max(1.0 - damage_history[i], 1e-12)
        Y[i] = strain_energy_history[i] / denom

    dissipated = np.sum(Y * dD) * element_volume
    # 裂纹面积近似: 单元截面
    A_crack = element_volume ** (2.0 / 3.0)
    G = dissipated / max(A_crack, 1e-12)
    return G


def extract_fracture_clusters(fractured_flags, connectivity):
    """
    通过连通性分析提取断裂簇

    输入:
        fractured_flags: (N_elem,) bool 数组
        connectivity: (N_elem, 4) 单元节点连接
    输出:
        clusters: list of list，每个子列表是一个断裂簇的单元索引
    """
    n_elem = len(fractured_flags)
    visited = np.zeros(n_elem, dtype=bool)
    clusters = []

    # 构建单元邻接图（共享面即邻接）
    adjacency = [set() for _ in range(n_elem)]
    for i in range(n_elem):
        if not fractured_flags[i]:
            continue
        nodes_i = set(connectivity[i])
        for j in range(i + 1, n_elem):
            if not fractured_flags[j]:
                continue
            nodes_j = set(connectivity[j])
            if len(nodes_i & nodes_j) >= 3:  # 共享面（3个节点）
                adjacency[i].add(j)
                adjacency[j].add(i)

    # DFS 找连通分量
    for i in range(n_elem):
        if not fractured_flags[i] or visited[i]:
            continue
        stack = [i]
        cluster = []
        visited[i] = True
        while stack:
            cur = stack.pop()
            cluster.append(cur)
            for nb in adjacency[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        clusters.append(cluster)

    return clusters


def critical_damage_criterion(equivalent_stress, yield_stress, damage_evolution_rate,
                               material_fracture_toughness, element_size):
    """
    综合断裂准则

    数学:
      损伤启动条件: σ_eq >= σ_y
      损伤演化条件: dD/dt > 0
      完全断裂条件: D >= D_c AND G >= G_c

    其中临界损伤阈值:
      D_c = 1 - (G_c / (σ_y * h))
      h 为特征单元尺寸
    """
    sigma_eq = float(equivalent_stress)
    sigma_y = float(yield_stress)
    dD_dt = float(damage_evolution_rate)
    G_c = float(material_fracture_toughness)
    h = float(element_size)

    if sigma_eq < sigma_y or dD_dt <= 1e-15:
        return False, 0.0

    # 近似临界损伤
    D_c = 1.0 - G_c / max(sigma_y * h, 1e-12)
    D_c = np.clip(D_c, 0.1, 0.99)

    return True, D_c
