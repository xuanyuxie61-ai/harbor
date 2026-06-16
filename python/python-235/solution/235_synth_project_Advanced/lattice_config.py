"""
lattice_config.py — 1+1维格点配置模块
======================================
种子项目映射:
  270_dfield9 (方向场/ODE) → 格点参数化与坐标生成
  002_advection_pde (持久参数) → 格点参数的持久化存储
  508_hb_to_mm (稀疏格式转换) → 稀疏邻接结构构建

物理: 1+1D标量场论格点离散化
  作用量: S = ∫dt dx [½(∂φ/∂t)² - ½(∂φ/∂x)² - V(φ)]
  V(φ) = ½m²φ² + λ/4! φ⁴ + μ³φ (BSM tadpole)
"""
import numpy as np


class LatticeConfig:
    """1+1维格点配置."""

    def __init__(self, N_x=256, h=0.1, dt=0.05, N_t=200,
                 mass=1.0, lam=0.1, mu3=0.0, boundary="periodic"):
        if N_x < 8:
            raise ValueError(f"N_x={N_x}<8, 高阶差分需要至少8点")
        if h <= 0 or dt <= 0:
            raise ValueError("h, dt 必须为正")

        self.N_x = N_x
        self.h = h
        self.N_t = N_t
        self.mass = mass
        self.lam = lam
        self.mu3 = mu3
        self.boundary = boundary
        self.L = N_x * h

        # CFL稳定性: dt <= h/sqrt(2) for 4th-order FD
        self.dt_max = h / np.sqrt(2.0)
        self.dt = min(dt, 0.9 * self.dt_max)
        self.T = N_t * self.dt

        self.x = np.arange(N_x) * h
        self.t = np.arange(N_t + 1) * self.dt
        self.k = np.fft.fftfreq(N_x, d=h) * 2 * np.pi

        self._build_sparse_indices()

    def _build_sparse_indices(self):
        """构建4阶差分模板的稀疏索引(CSR-like)."""
        bw = 2
        self.row_ptr = np.arange(self.N_x + 1) * (2 * bw + 1)
        col = np.zeros(self.N_x * (2 * bw + 1), dtype=np.int32)
        for i in range(self.N_x):
            for j, off in enumerate(range(-bw, bw + 1)):
                col[i * (2 * bw + 1) + j] = (i + off) % self.N_x
        self.col_idx = col
        # 4阶中心差分系数: [-1, 16, -30, 16, -1]/(12h²)
        self.fd4_coeff = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * self.h**2)

    def dispersion(self, k=None):
        """色散关系: ω²(k) = k̃²(k) + m²."""
        if k is None:
            k = self.k
        kh = k * self.h
        k_tilde_sq = (30.0 - 32.0 * np.cos(kh) + 2.0 * np.cos(2 * kh)) / (12.0 * self.h**2)
        return np.sqrt(np.maximum(k_tilde_sq, 0.0) + self.mass**2)

    def summary(self):
        return (f"Lattice: N_x={self.N_x}, h={self.h}, dt={self.dt:.4f}, "
                f"m={self.mass}, λ={self.lam}, μ³={self.mu3}, "
                f"CFL_dt_max={self.dt_max:.4f}")
