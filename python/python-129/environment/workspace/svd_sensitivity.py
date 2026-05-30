
import numpy as np


class ParameterSensitivitySVD:

    def __init__(self, param_names, response_names):
        self.param_names = param_names
        self.response_names = response_names
        self.n_params = len(param_names)
        self.n_responses = len(response_names)

    def compute_sensitivity_matrix(self, model_func, p0, delta_frac=0.01):
        p0 = np.asarray(p0, dtype=float)
        if p0.shape[0] != self.n_params:
            raise ValueError("p0 长度必须等于参数数量")

        R0 = np.asarray(model_func(p0), dtype=float)
        if R0.shape[0] != self.n_responses:
            raise ValueError("model_func 返回值长度必须等于响应指标数量")

        S = np.zeros((self.n_responses, self.n_params))
        for j in range(self.n_params):
            h = delta_frac * max(abs(p0[j]), 1e-12)
            p_pert = p0.copy()
            p_pert[j] += h
            R_pert = np.asarray(model_func(p_pert), dtype=float)
            S[:, j] = (R_pert - R0) / h
        return S

    def svd_decompose(self, S):
        S = np.asarray(S, dtype=float)
        U, sigma, Vt = np.linalg.svd(S, full_matrices=False)

        total_var = np.sum(sigma ** 2)
        if total_var > 0:
            cum_var = np.cumsum(sigma ** 2) / total_var
        else:
            cum_var = np.zeros_like(sigma)
        return U, sigma, Vt, cum_var

    def principal_parameter_directions(self, Vt, n_components=3):
        directions = []
        for k in range(min(n_components, Vt.shape[0])):
            v_k = Vt[k, :]

            ranked = sorted(enumerate(v_k), key=lambda x: abs(x[1]), reverse=True)
            top = [(self.param_names[i], float(val)) for i, val in ranked[:5]]
            directions.append({
                "component": k + 1,
                "top_params": top
            })
        return directions

    def reduced_basis_fit(self, S, target_response, p0, n_components=3):
        U, sigma, Vt, _ = self.svd_decompose(S)
        r = min(n_components, len(sigma))
        Ur = U[:, :r]
        Sr_inv = np.diag(1.0 / (sigma[:r] + 1e-30))
        Vr = Vt[:r, :].T

        residual = target_response - S @ p0
        alpha = Sr_inv @ Ur.T @ residual
        p_opt = p0 + Vr @ alpha

        p_opt = np.maximum(p_opt, 1e-12)
        return p_opt, alpha


def demo_svd_sensitivity():
    param_names = [
        "k_cat_TF_VIIa_IX", "k_cat_IXa_X", "k_cat_Xa_II",
        "k_inact_ATIII", "k_TFPI_inact", "k_APC_inact_Va",
        "k_polymerization", "k_fibrin_lysis", "k_PLT_act",
        "D_fibrin", "k_clear", "k_plasminogen_act"
    ]
    response_names = ["Peak_IIa", "TFT_50_percent", "Fibrin_yield", "Clot_stability"]

    analyzer = ParameterSensitivitySVD(param_names, response_names)


    def mock_model(p):

        r1 = 100.0 * p[2] / (p[3] + 0.1)
        r2 = 60.0 / (p[0] + 0.01) + 20.0 * p[4]
        r3 = 50.0 * p[6] / (p[7] + 0.1)
        r4 = 80.0 * p[6] - 40.0 * p[7] + 10.0 * p[8]
        return np.array([r1, r2, r3, r4])

    p0 = np.array([1.2, 6.5, 25.0, 0.003, 0.015, 0.05,
                   0.5, 0.008, 0.1, 0.025, 0.001, 0.02])

    S = analyzer.compute_sensitivity_matrix(mock_model, p0, delta_frac=0.01)
    U, sigma, Vt, cum_var = analyzer.svd_decompose(S)

    print("=" * 60)
    print("血凝级联参数敏感性SVD分析")
    print("=" * 60)
    print(f"奇异值: {sigma}")
    print(f"累积方差解释: {cum_var}")

    directions = analyzer.principal_parameter_directions(Vt, n_components=3)
    for d in directions:
        print(f"\n主成分 {d['component']} 主导参数:")
        for name, val in d['top_params'][:3]:
            print(f"  {name:25s}: {val: .4f}")

    target = np.array([200.0, 100.0, 100.0, 50.0])
    p_opt, alpha = analyzer.reduced_basis_fit(S, target, p0, n_components=3)
    print(f"\n低维拟合系数 α: {alpha}")
    print(f"拟合后响应: {mock_model(p_opt)}")
    return analyzer, S, sigma


if __name__ == "__main__":
    demo_svd_sensitivity()
