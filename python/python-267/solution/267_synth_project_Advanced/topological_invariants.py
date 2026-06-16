"""
topological_invariants.py — 拓扑不变量计算
============================================

本模块实现拓扑绝缘体的关键拓扑不变量计算:

1. **Chern 数** (陈数): 表征量子反常霍尔效应 (QAHE)
2. **Z₂ 不变量**: 表征量子自旋霍尔效应 (QSHE) / 时间反演拓扑绝缘体
3. **Berry 曲率**: 拓扑不变量的微分形式
4. **Wilson loop / Wannier 中心**: Z₂ 不变量的现代计算方法

物理公式
--------
**Berry 联络:**
    Aₙ(k) = -i ⟨uₙ(k)| ∇ₖ |uₙ(k)⟩

**Berry 曲率:**
    Fₙ(k) = ∇ₖ × Aₙ(k)
    = -2 Im Σ_{m≠n} ⟨uₙ|∂H/∂kx|uₘ⟩⟨uₘ|∂H/∂ky|uₙ⟩ / (Eₘ-Eₙ)²

**Chern 数:**
    Cₙ = (1/2π) ∫∫_{BZ} Fₙ(kx, ky) dkx dky ∈ ℤ

**Z₂ 不变量 (Fu-Kane 方法):**
    (-1)^ν = Π_{i=1}^{4} δ_{TRIM_i}
    其中 δ_{TRIM} = Π_{m=1}^{N_occ} ξ_{2m}(Λ_i)
    ξ = ±1 为时间反演算子的本征值

**离散化 Fukui-Hatsugai-Suzuki 方法:**
    在 BZ 网格上定义:
    U₁(k) = det⟨uₙ(k)|uₙ(k+δkx)⟩ / |det|
    U₂(k) = det⟨uₙ(k)|uₙ(k+δky)⟩ / |det|

    F₁₂(k) = ln[U₁(k) U₂(k+δkx) U₁(k+δky)⁻¹ U₂(k)⁻¹]
    C = (1/2πi) Σ_k F₁₂(k)

来源映射
--------
- 1019_olajuwonOG: Berry 曲率计算类比于结构因子 S(q) 的复指数求和
- 299_disk01_integrals: BZ 积分的精确求积规则
- 1345_triangulation_plot: BZ 三角网格剖分
"""

import numpy as np
from typing import Tuple, Dict, List
from bhz_hamiltonian import BHZParameters, bhz_h_k


def berry_curvature_kubo(kx: float, ky: float, params: BHZParameters,
                         band_index: int = 0, eta: float = 1e-6
                         ) -> float:
    """
    Kubo 公式计算单 k 点的 Berry 曲率。

    Fₙ(k) = -2 Im Σ_{m≠n} <uₙ|∂H/∂kx|uₘ><uₘ|∂H/∂ky|uₙ> / (Eₘ-Eₙ)²

    对 BHZ 模型, 解析结果为:
        F(k) = ± (1/2) (ABM₀) / (M²+A²k²)^{3/2}
    (对导带取负号, 价带取正号, 忽略 D 项)

    Parameters
    ----------
    kx, ky : float
    params : BHZParameters
    band_index : int
        能带索引 (0=最低价带)
    eta : float
        正则化参数

    Returns
    -------
    F : float
        Berry 曲率 (nm²)
    """
    k2 = kx * kx + ky * ky
    M_k = params.M0 - params.B * k2
    gap_half = np.sqrt(M_k ** 2 + params.A ** 2 * k2)

    if gap_half < 1e-15:
        return 0.0

    # 数值计算: 先对角化 H(k)
    H = bhz_h_k(kx, ky, params)
    evals, evecs = np.linalg.eigh(H)

    # 数值导数 ∂H/∂kx, ∂H/∂ky (中心差分)
    dk = 1e-5
    H_px = bhz_h_k(kx + dk, ky, params)
    H_mx = bhz_h_k(kx - dk, ky, params)
    H_py = bhz_h_k(kx, ky + dk, params)
    H_my = bhz_h_k(kx, ky - dk, params)

    dHdkx = (H_px - H_mx) / (2 * dk)
    dHdky = (H_py - H_my) / (2 * dk)

    n = band_index
    F = 0.0
    for m in range(4):
        if m == n:
            continue
        dE = evals[m] - evals[n]
        if abs(dE) < eta:
            continue

        # <uₙ|∂H/∂kx|uₘ>
        vx_nm = evecs[:, n].conj() @ dHdkx @ evecs[:, m]
        # <uₘ|∂H/∂ky|uₙ>
        vy_mn = evecs[:, m].conj() @ dHdky @ evecs[:, n]

        F += -2.0 * np.imag(vx_nm * vy_mn) / (dE ** 2)

    return F


def berry_curvature_analytic(kx: float, ky: float,
                             params: BHZParameters) -> float:
    """
    BHZ 模型价带 Berry 曲率的解析表达式。

    对简化的 BHZ (D=0):
        F(k) = -AB M(k) / (2 (M(k)² + A²k²)^{3/2})

    这是 Dirac 模型的标准结果。

    Parameters
    ----------
    kx, ky : float
    params : BHZParameters

    Returns
    -------
    F : float
    """
    k2 = kx * kx + ky * ky
    M_k = params.M0 - params.B * k2
    denominator = 2.0 * (M_k ** 2 + params.A ** 2 * k2) ** 1.5

    if abs(denominator) < 1e-30:
        return 0.0

    F = -params.A * params.B * M_k / denominator
    return F


def chern_number_fhs(params: BHZParameters, Nk: int = 50,
                     band_index: int = 0) -> Dict:
    """
    Fukui-Hatsugai-Suzuki 方法计算 Chern 数。

    在 BZ 上均匀取 Nk×Nk 网格, 计算 Berry 相位围绕每个小 plaquette 的贡献。

    步骤:
    1. 在每个 k 点对角化 H(k), 得到 |uₙ(k)⟩
    2. 计算链路变量:
       U₁(k) = ⟨uₙ(k)|uₙ(k+δ₁)⟩ / |⟨uₙ(k)|uₙ(k+δ₁)⟩|
       U₂(k) = ⟨uₙ(k)|uₙ(k+δ₂)⟩ / |⟨uₙ(k)|uₙ(k+δ₂)⟩|
    3. Plaquette 相位:
       F₁₂(k) = ln[U₁(k) U₂(k+δ₁) U₁(k+δ₂)⁻¹ U₂(k)⁻¹]
    4. Chern 数:
       C = (1/2πi) Σ_k F₁₂(k)

    该方法的优点: 结果严格为整数, 不依赖网格密度 (只要网格足够细)。

    Parameters
    ----------
    params : BHZParameters
    Nk : int
        BZ 网格点数
    band_index : int
        能带索引

    Returns
    -------
    result : dict
    """
    # BZ 范围 (简化为 [-π/a, π/a]²)
    a = 1.0  # 晶格常数 (nm), 归一化
    kx_arr = np.linspace(-np.pi / a, np.pi / a, Nk, endpoint=False)
    ky_arr = np.linspace(-np.pi / a, np.pi / a, Nk, endpoint=False)
    dkx = kx_arr[1] - kx_arr[0] if Nk > 1 else 2 * np.pi / a
    dky = ky_arr[1] - ky_arr[0] if Nk > 1 else 2 * np.pi / a

    # 存储本征矢
    # evecs_arr[ik, il, :, n] = 第 n 个本征矢在 (kx[ik], ky[il]) 处
    evecs_arr = np.zeros((Nk, Nk, 4, 4), dtype=np.complex128)
    evals_arr = np.zeros((Nk, Nk, 4))

    for ik in range(Nk):
        for il in range(Nk):
            H = bhz_h_k(kx_arr[ik], ky_arr[il], params)
            evals, evecs = np.linalg.eigh(H)
            evals_arr[ik, il] = evals
            evecs_arr[ik, il] = evecs

    # 计算链路变量和 plaquette 相位
    n = band_index
    total_phase = 0.0

    for ik in range(Nk):
        for il in range(Nk):
            ik1 = (ik + 1) % Nk
            il1 = (il + 1) % Nk

            # |uₙ(k)⟩
            u_00 = evecs_arr[ik, il, :, n]
            u_10 = evecs_arr[ik1, il, :, n]
            u_01 = evecs_arr[ik, il1, :, n]
            u_11 = evecs_arr[ik1, il1, :, n]

            # 链路变量
            U1_00 = np.vdot(u_00, u_10)  # ⟨u(k)|u(k+δx)⟩
            U2_00 = np.vdot(u_00, u_01)  # ⟨u(k)|u(k+δy)⟩
            U1_01 = np.vdot(u_01, u_11)  # ⟨u(k+δy)|u(k+δx+δy)⟩
            U2_10 = np.vdot(u_10, u_11)  # ⟨u(k+δx)|u(k+δx+δy)⟩

            # 归一化
            U1_00 /= max(abs(U1_00), 1e-30)
            U2_00 /= max(abs(U2_00), 1e-30)
            U1_01 /= max(abs(U1_01), 1e-30)
            U2_10 /= max(abs(U2_10), 1e-30)

            # Plaquette: F = ln(U₁·U₂(k+δx)·U₁(k+δy)⁻¹·U₂⁻¹)
            F_plaq = U1_00 * U2_10 / (U1_01 * U2_00)
            phase = np.angle(F_plaq)  # 取主值 ∈ (-π, π]

            total_phase += phase

    chern = total_phase / (2.0 * np.pi)

    # 数值 Berry 曲率积分 (对比)
    F_sum = 0.0
    for ik in range(Nk):
        for il in range(Nk):
            F_val = berry_curvature_kubo(kx_arr[ik], ky_arr[il], params,
                                         band_index)
            F_sum += F_val * dkx * dky
    chern_from_curvature = F_sum / (2.0 * np.pi)

    return {
        'chern_fhs': chern,
        'chern_fhs_rounded': int(np.round(chern)),
        'chern_from_curvature_integral': chern_from_curvature,
        'Nk_grid': Nk,
        'band_index': band_index,
        'total_phase': total_phase,
        'is_integer': abs(chern - np.round(chern)) < 0.01
    }


def z2_invariant_parity(params: BHZParameters) -> Dict:
    """
    通过 TRIM (时间反演不变动量) 点的宇称计算 Z₂ 不变量。

    Fu-Kane 方法:
        (-1)^ν = Π_{i=1}^{4} δ_i
        δ_i = Π_{m ∈ occ} ξ_{2m}(Λ_i)

    TRIM 点: Γ=(0,0), X=(π,0), Y=(0,π), M=(π,π)

    在 TRIM 点, 时间反演算子 T² = -1 的 Kramers 对简并,
    每个 Kramers 对有确定的宇称 ξ = ±1。

    对 BHZ 模型:
        在 Γ 点: h(0) = diag(M0, -M0, M0, -M0)
        占据带 (E<0) 的宇称由 M0 的符号决定

    Parameters
    ----------
    params : BHZParameters

    Returns
    -------
    result : dict
    """
    # TRIM 点
    trims = {
        'Gamma': (0.0, 0.0),
        'X': (np.pi, 0.0),
        'Y': (0.0, np.pi),
        'M': (np.pi, np.pi),
    }

    # 对每个 TRIM 点, 计算占据带的宇称
    deltas = {}
    for name, (kx, ky) in trims.items():
        H = bhz_h_k(kx, ky, params)
        evals, evecs = np.linalg.eigh(H)

        # 占据带 (能量 < C, 即相对带中心为负)
        E_ref = params.C
        occupied = evals < E_ref

        # 在 TRIM 点, 时间反演对称性保证 Kramers 简并
        # 占据带应成对出现 (Kramers pairs)
        # 宇称从 h(k) 的 d₃ 分量符号读取

        k2 = kx * kx + ky * ky
        M_k = params.M0 - params.B * k2

        # 简化的宇称: ξ = sign(M_k) 对每对占据带
        # 对 BHZ, δ = sign(M_k)² = 1 (因为两对简并带贡献相同)
        # 但实际上 δ = sign(M_k) 对每一对, 两对相同
        delta = 1 if M_k > 0 else -1
        deltas[name] = delta

    # Z₂ 不变量
    product_delta = 1
    for d in deltas.values():
        product_delta *= d

    z2 = 0 if product_delta > 0 else 1  # (-1)^ν, ν=0 trivial, ν=1 topological

    return {
        'z2_invariant': z2,
        'topological': z2 == 1,
        'trimming_parities': deltas,
        'product_delta': product_delta,
        'M0': params.M0,
        'is_topological': params.is_topological
    }


def berry_curvature_map(params: BHZParameters, Nk: int = 30,
                        band_index: int = 0) -> Dict:
    """
    计算 BZ 上的 Berry 曲率分布图 (数值形式, 非可视化)。

    Parameters
    ----------
    params : BHZParameters
    Nk : int
    band_index : int

    Returns
    -------
    result : dict
    """
    kx_arr = np.linspace(-np.pi, np.pi, Nk)
    ky_arr = np.linspace(-np.pi, np.pi, Nk)
    dkx = kx_arr[1] - kx_arr[0]
    dky = ky_arr[1] - ky_arr[0]

    F_map = np.zeros((Nk, Nk))
    F_analytic = np.zeros((Nk, Nk))

    for ik in range(Nk):
        for il in range(Nk):
            F_map[ik, il] = berry_curvature_kubo(
                kx_arr[ik], ky_arr[il], params, band_index
            )
            F_analytic[ik, il] = berry_curvature_analytic(
                kx_arr[ik], ky_arr[il], params
            )

    # 积分验证
    chern_numerical = np.sum(F_map) * dkx * dky / (2 * np.pi)
    chern_analytic = np.sum(F_analytic) * dkx * dky / (2 * np.pi)

    return {
        'kx': kx_arr,
        'ky': ky_arr,
        'F_numerical': F_map,
        'F_analytic': F_analytic,
        'chern_numerical': chern_numerical,
        'chern_analytic': chern_analytic,
        'max_curvature': float(np.max(np.abs(F_map))),
        'Nk': Nk
    }


def topological_phase_diagram(params_list: List[BHZParameters],
                              label: str = 'M0_scan') -> Dict:
    """
    扫描参数空间, 绘制拓扑相图。

    Parameters
    ----------
    params_list : list of BHZParameters
    label : str

    Returns
    -------
    result : dict
    """
    results = []
    for p in params_list:
        z2 = z2_invariant_parity(p)
        results.append({
            'M0': p.M0,
            'B': p.B,
            'z2': z2['z2_invariant'],
            'topological': z2['topological'],
            'bulk_gap': p.bulk_gap,
            'M0_B_ratio': p.M0 / p.B if p.B != 0 else np.inf
        })

    # 找相变点
    phase_transitions = []
    for i in range(1, len(results)):
        if results[i]['z2'] != results[i-1]['z2']:
            # 线性插值找精确相变点
            M0_1 = results[i-1]['M0']
            M0_2 = results[i]['M0']
            M0_c = (M0_1 + M0_2) / 2.0  # 近似
            phase_transitions.append(M0_c)

    return {
        'label': label,
        'points': results,
        'phase_transitions': phase_transitions,
        'n_topological': sum(1 for r in results if r['topological']),
        'n_trivial': sum(1 for r in results if not r['topological'])
    }


def topological_invariants_report(params: BHZParameters,
                                  Nk: int = 20) -> str:
    """生成拓扑不变量计算的完整报告。"""
    lines = []
    lines.append("=" * 60)
    lines.append("拓扑不变量计算报告")
    lines.append("=" * 60)
    lines.append(f"BHZ 参数: {params}")
    lines.append(f"拓扑判据 M0/B = {params.M0/B:.4f}" if (B:=params.B) != 0 else "")
    lines.append(f"体带隙 = {params.bulk_gap:.6f} eV")
    lines.append(f"拓扑非平庸: {params.is_topological}")

    # Z₂
    z2 = z2_invariant_parity(params)
    lines.append(f"\n--- Z₂ 不变量 ---")
    lines.append(f"  ν = {z2['z2_invariant']}")
    lines.append(f"  拓扑相: {z2['topological']}")
    for name, delta in z2['trimming_parities'].items():
        lines.append(f"  δ({name}) = {delta:+d}")

    # Chern 数 (使用较小网格以节省时间)
    lines.append(f"\n--- Chern 数 (FHS 方法, Nk={Nk}) ---")
    for band in range(2):  # 只算占据带
        chern = chern_number_fhs(params, Nk=Nk, band_index=band)
        lines.append(f"  能带 {band}: C = {chern['chern_fhs']:.4f} "
                     f"(取整 = {chern['chern_fhs_rounded']})")

    # Berry 曲率
    lines.append(f"\n--- Berry 曲率统计 ---")
    bc = berry_curvature_map(params, Nk=min(Nk, 15))
    lines.append(f"  最大曲率: {bc['max_curvature']:.6e} nm²")
    lines.append(f"  Chern 数 (曲率积分): {bc['chern_numerical']:.4f}")

    return "\n".join(lines)
