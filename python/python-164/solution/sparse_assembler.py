"""
sparse_assembler.py
催化剂层离散化稀疏矩阵组装模块

基于 hb_to_mm (508) 与 r8bb (969) 改造
用于组装 PEM 燃料电池催化剂层多物理场耦合模型的离散化稀疏矩阵。

核心公式:
  有限差分/有限体积离散化后，系统方程可写为:
    K * u = f
  
  其中 K 为稀疏矩阵，具有以下块结构 (边界带状):
    
    K = [ K11  K12 ]
        [ K21  K22 ]
  
  K11 对应催化剂层内部网格点的三对角/带状部分 (扩散-反应算子),
  K12 对应边界耦合条件,
  K21 对应界面通量约束,
  K22 对应外部电路方程。
  
  使用边界带状 (Border-Banded) 格式存储以节省内存。
  
  此外支持 Harwell-Boeing 风格的稀疏矩阵元数据管理。
"""

import numpy as np


class SparseAssembler:
    """
    催化剂层离散化稀疏矩阵组装器。
    
    组装形式:
        [ D - L  0  ] [C]   [b1]
        [ -L  D -L  ] [T] = [b2]
        [  0  -L  D ] [E]   [b3]
    
    其中 D, L 分别为对角线和非对角线系数。
    """
    
    def __init__(self, n_interior, n_boundary=2):
        """
        参数:
            n_interior: 内部网格点数
            n_boundary: 边界/耦合方程数
        """
        if n_interior < 1 or n_boundary < 0:
            raise ValueError("参数无效")
        
        self.n1 = n_interior
        self.n2 = n_boundary
        self.n_total = n_interior + n_boundary
    
    def assemble_diffusion_reaction_matrix(self, D_coeff, k_rxn, dx, bc_type='dirichlet_neumann'):
        """
        组装一维稳态扩散-反应离散矩阵。
        
        参数:
            D_coeff: 扩散系数 [m^2/s]
            k_rxn: 反应速率常数 [1/s]
            dx: 网格间距 [m]
            bc_type: 边界条件类型
        
        返回:
            A_full: 完整稠密矩阵 (用于验证)
            A_band: 带状存储矩阵
        """
        n = self.n1
        A_full = np.zeros((n, n))
        
        coeff = D_coeff / (dx * dx)
        
        # 内部点
        for i in range(1, n - 1):
            A_full[i, i - 1] = -coeff
            A_full[i, i] = 2.0 * coeff + k_rxn
            A_full[i, i + 1] = -coeff
        
        # 边界处理
        if bc_type == 'dirichlet_neumann':
            # 左边界 Dirichlet
            A_full[0, 0] = 1.0
            # 右边界 Neumann: (C_N - C_{N-1})/dx = 0
            A_full[n - 1, n - 2] = -1.0
            A_full[n - 1, n - 1] = 1.0
        elif bc_type == 'dirichlet_dirichlet':
            A_full[0, 0] = 1.0
            A_full[n - 1, n - 1] = 1.0
        elif bc_type == 'robin_robin':
            # Robin 条件: a*C + b*dC/dx = c
            A_full[0, 0] = 1.0 + coeff * dx  # 简化处理
            A_full[0, 1] = -coeff * dx
            A_full[n - 1, n - 2] = -coeff * dx
            A_full[n - 1, n - 1] = 1.0 + coeff * dx
        else:
            raise ValueError(f"未知边界条件类型: {bc_type}")
        
        # 转换为带状存储
        ml, mu = 1, 1
        m_band = 2 * ml + mu + 1
        A_band = np.zeros((m_band, n))
        
        for j in range(n):
            for i in range(max(0, j - mu), min(n, j + ml + 1)):
                row_in_band = i - j + ml + mu
                A_band[row_in_band, j] = A_full[i, j]
        
        return A_full, A_band
    
    def assemble_coupled_system(self, D_eff, k_rxn, dx, 
                                 R_ct, j0, alpha_a, alpha_c, n_e, T):
        """
        组装扩散-反应-电化学耦合系统的边界带状矩阵。
        
        扩展系统包含:
          - n1 个内部网格点 (扩散-反应)
          - n2 个边界耦合方程 (电化学动力学)
        
        基于 r8bb 格式存储。
        """
        n1 = self.n1
        n2 = self.n2
        
        # A1: 带状部分 (扩散-反应)
        ml, mu = 1, 1
        nband = (2 * ml + mu + 1) * n1
        
        # 总存储大小
        total_size = nband + 2 * n1 * n2 + n2 * n2
        A_bb = np.zeros(total_size)
        
        # 填充 A1 (带状)
        _, A1_band = self.assemble_diffusion_reaction_matrix(D_eff, k_rxn, dx)
        A_bb[0:nband] = self._r8gb_to_r8vec(n1, n1, ml, mu, A1_band)
        
        # 填充 A2 (内部点与边界耦合)
        # 简化: 最后一个内部点与边界条件耦合
        for j in range(n2):
            for i in range(n1):
                idx = nband + j * n1 + i
                if i == n1 - 1 and j == 0:
                    A_bb[idx] = -D_eff / dx  # 通量耦合
                else:
                    A_bb[idx] = 0.0
        
        # 填充 A3 (边界与内部耦合)
        for j in range(n1):
            for i in range(n2):
                idx = nband + n1 * n2 + j * n2 + i
                if j == n1 - 1 and i == 0:
                    A_bb[idx] = -1.0
                else:
                    A_bb[idx] = 0.0
        
        # 填充 A4 (边界方程)
        # 电化学边界: eta 方程
        for j in range(n2):
            for i in range(n2):
                idx = nband + 2 * n1 * n2 + j * n2 + i
                if i == j:
                    A_bb[idx] = 1.0
                else:
                    A_bb[idx] = 0.0
        
        return A_bb, ml, mu
    
    def _r8gb_to_r8vec(self, n, m, ml, mu, a_gb):
        """
        将 R8GB 带状矩阵转换为向量存储。
        """
        a_vec = np.zeros((2 * ml + mu + 1) * n)
        for j in range(n):
            for i in range(max(0, j - mu), min(m, j + ml + 1)):
                k = i - j + ml + mu
                idx = k + j * (2 * ml + mu + 1)
                if idx < len(a_vec):
                    a_vec[idx] = a_gb[k, j]
        return a_vec
    
    def extract_diagonal(self, A_full):
        """
        提取矩阵对角线，用于预处理和收敛性分析。
        """
        n = min(A_full.shape)
        diag = np.zeros(n)
        for i in range(n):
            diag[i] = A_full[i, i]
        return diag
    
    def check_diagonal_dominance(self, A_full):
        """
        检查矩阵是否严格对角占优。
        
        对于三对角矩阵，这是循环约化法收敛的充分条件。
        """
        n = A_full.shape[0]
        is_dominant = True
        min_ratio = float('inf')
        
        for i in range(n):
            diag = abs(A_full[i, i])
            off_diag = np.sum(np.abs(A_full[i, :])) - diag
            
            if diag <= off_diag:
                is_dominant = False
            
            if off_diag > 0:
                ratio = diag / off_diag
                min_ratio = min(min_ratio, ratio)
            else:
                min_ratio = min(min_ratio, float('inf'))
        
        return is_dominant, min_ratio


def harwell_boeing_metadata(nrow, ncol, nnzero, title="CCL_Sparse", key="CCL_01"):
    """
    生成 Harwell-Boeing 风格的稀疏矩阵元数据。
    
    基于 hb_to_mm (508) 的格式概念改造。
    """
    metadata = {
        'title': title,
        'key': key,
        'nrow': nrow,
        'ncol': ncol,
        'nnzero': nnzero,
        'matrix_type': 'RUA',  # Real Unsymmetric Assembled
        'ptrfmt': '(8I10)',
        'indfmt': '(8I10)',
        'valfmt': '(4D20.13)',
    }
    return metadata


if __name__ == "__main__":
    assembler = SparseAssembler(n_interior=20, n_boundary=2)
    A_full, A_band = assembler.assemble_diffusion_reaction_matrix(1e-9, 100.0, 1e-7)
    is_dd, ratio = assembler.check_diagonal_dominance(A_full)
    print(f"矩阵维度: {A_full.shape}, 严格对角占优: {is_dd}, 最小比值: {ratio:.4f}")
