
import numpy as np


class SparseAssembler:
    
    def __init__(self, n_interior, n_boundary=2):
        if n_interior < 1 or n_boundary < 0:
            raise ValueError("参数无效")
        
        self.n1 = n_interior
        self.n2 = n_boundary
        self.n_total = n_interior + n_boundary
    
    def assemble_diffusion_reaction_matrix(self, D_coeff, k_rxn, dx, bc_type='dirichlet_neumann'):
        n = self.n1
        A_full = np.zeros((n, n))
        
        coeff = D_coeff / (dx * dx)
        

        for i in range(1, n - 1):
            A_full[i, i - 1] = -coeff
            A_full[i, i] = 2.0 * coeff + k_rxn
            A_full[i, i + 1] = -coeff
        

        if bc_type == 'dirichlet_neumann':

            A_full[0, 0] = 1.0

            A_full[n - 1, n - 2] = -1.0
            A_full[n - 1, n - 1] = 1.0
        elif bc_type == 'dirichlet_dirichlet':
            A_full[0, 0] = 1.0
            A_full[n - 1, n - 1] = 1.0
        elif bc_type == 'robin_robin':

            A_full[0, 0] = 1.0 + coeff * dx
            A_full[0, 1] = -coeff * dx
            A_full[n - 1, n - 2] = -coeff * dx
            A_full[n - 1, n - 1] = 1.0 + coeff * dx
        else:
            raise ValueError(f"未知边界条件类型: {bc_type}")
        

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
        n1 = self.n1
        n2 = self.n2
        

        ml, mu = 1, 1
        nband = (2 * ml + mu + 1) * n1
        

        total_size = nband + 2 * n1 * n2 + n2 * n2
        A_bb = np.zeros(total_size)
        

        _, A1_band = self.assemble_diffusion_reaction_matrix(D_eff, k_rxn, dx)
        A_bb[0:nband] = self._r8gb_to_r8vec(n1, n1, ml, mu, A1_band)
        


        for j in range(n2):
            for i in range(n1):
                idx = nband + j * n1 + i
                if i == n1 - 1 and j == 0:
                    A_bb[idx] = -D_eff / dx
                else:
                    A_bb[idx] = 0.0
        

        for j in range(n1):
            for i in range(n2):
                idx = nband + n1 * n2 + j * n2 + i
                if j == n1 - 1 and i == 0:
                    A_bb[idx] = -1.0
                else:
                    A_bb[idx] = 0.0
        


        for j in range(n2):
            for i in range(n2):
                idx = nband + 2 * n1 * n2 + j * n2 + i
                if i == j:
                    A_bb[idx] = 1.0
                else:
                    A_bb[idx] = 0.0
        
        return A_bb, ml, mu
    
    def _r8gb_to_r8vec(self, n, m, ml, mu, a_gb):
        a_vec = np.zeros((2 * ml + mu + 1) * n)
        for j in range(n):
            for i in range(max(0, j - mu), min(m, j + ml + 1)):
                k = i - j + ml + mu
                idx = k + j * (2 * ml + mu + 1)
                if idx < len(a_vec):
                    a_vec[idx] = a_gb[k, j]
        return a_vec
    
    def extract_diagonal(self, A_full):
        n = min(A_full.shape)
        diag = np.zeros(n)
        for i in range(n):
            diag[i] = A_full[i, i]
        return diag
    
    def check_diagonal_dominance(self, A_full):
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
    metadata = {
        'title': title,
        'key': key,
        'nrow': nrow,
        'ncol': ncol,
        'nnzero': nnzero,
        'matrix_type': 'RUA',
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
