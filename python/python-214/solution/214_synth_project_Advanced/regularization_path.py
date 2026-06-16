"""
regularization_path.py — 正则化路径追踪与 Newton-Maehly 延拓
=============================================================
来源项目映射:
  - 801_newton_maehly → Newton-Maehly 多项式求根与降阶 (deflation)

科学背景:
  LASSO 解路径 λ ↦ x*(λ) 是分段线性分段光滑曲线.
  本模块采用延拓 (continuation) 方法追踪整条路径:
    (1) 从 λ_max = ||A^T y||_∞ (此时 x = 0) 开始
    (2) 逐步递减 λ 至 λ_min
    (3) 每一步用 warm-start FISTA 求解
    (4) 用 Newton-Maehly deflation 检测路径上的分支点

  分支点检测:
    当支撑集 supp(x*(λ)) 在 λ_c 处发生突变时, 路径出现折点.
    定义 F(λ) = min_x 0.5||Ax-y||² + λ||x||_1 关于 λ 的次微分.
    用 Newton-Maehly 求 F'(λ) = 0 的点 (即路径转折点).

  Newton-Maehly deflation (源自 801):
    已知根 r_1, ..., r_k 后, 为求下一个根, 定义降阶函数:
        g(x) = f(x) / ∏_{i=1}^k (x - r_i)
    对 g 应用 Newton 法.

核心公式:
  λ_max = ||A^T y||_∞
  路径折点满足:  A_T^T (A_T x_T - y) + λ sign(x_T) = 0
  其中 T = supp(x*(λ)).
"""
import numpy as np
from l1_ista import fista_l1, soft_threshold, lipschitz_constant


# ----------------------------------------------------------------------
# 正则化路径追踪
# ----------------------------------------------------------------------
def compute_lambda_max(A, y):
    """λ_max = ||A^T y||_∞,  超过此值解恒为 0."""
    return float(np.max(np.abs(A.T @ y)))


def lambda_grid(lam_max, lam_min, n_lambda=50, log_scale=True):
    """生成 λ 网格 (对数或线性)."""
    if log_scale:
        return np.exp(np.linspace(np.log(lam_max), np.log(lam_min), n_lambda))
    return np.linspace(lam_max, lam_min, n_lambda)


def solve_regularization_path(A, y, n_lambda=40, lam_min_ratio=1e-3,
                              max_iter_per=1000, tol=1e-8):
    """追踪 LASSO 解路径.

    返回:
        lambdas    : (n_lambda,)
        solutions  : list of x*(λ)
        supports   : list of support sets
        bic_scores : BIC 准则值
    """
    lam_max = compute_lambda_max(A, y)
    lambdas = lambda_grid(lam_max, lam_max * lam_min_ratio, n_lambda)

    solutions = []
    supports = []
    bic_scores = []
    m, n = A.shape
    x_warm = np.zeros(n)

    for lam in lambdas:
        # warm-start: 用上一个解作为初始
        x_sol, _ = fista_l1(A, y, lam, x0=x_warm,
                            max_iter=max_iter_per, tol=tol)
        x_warm = x_sol.copy()
        sup = set(np.where(np.abs(x_sol) > 1e-10)[0])
        solutions.append(x_sol)
        supports.append(sup)
        # BIC: m log(RSS/m) + |sup| log(m)
        rss = np.sum((A @ x_sol - y) ** 2)
        bic = m * np.log(max(rss / m, 1e-16)) + len(sup) * np.log(m)
        bic_scores.append(bic)
    return lambdas, solutions, supports, bic_scores


def select_lambda_bic(lambdas, bic_scores):
    """BIC 准则选择最优 λ."""
    idx = int(np.argmin(bic_scores))
    return lambdas[idx], idx


def select_lambda_gcv(A, y, lambdas, solutions):
    """GCV 准则: GCV(λ) = ||y - A x_λ||² / (m - df_λ)².

    df_λ = rank(A_{supp}) 作为自由度近似.
    """
    m, n = A.shape
    gcv = []
    for lam, x in zip(lambdas, solutions):
        sup = np.where(np.abs(x) > 1e-10)[0]
        df = max(len(sup), 1)
        rss = np.sum((A @ x - y) ** 2)
        denom = max(m - df, 1)
        gcv.append(rss / (denom ** 2))
    idx = int(np.argmin(gcv))
    return lambdas[idx], idx


# ----------------------------------------------------------------------
# Newton-Maehly deflation (源自 801_newton_maehly)
# ----------------------------------------------------------------------
def poly_eval_and_deriv(coeffs, x):
    """多项式求值及其导数 (Horner 法).

    p(x) = c_0 + c_1 x + ... + c_n x^n
    """
    n = len(coeffs) - 1
    p = coeffs[-1]
    dp = 0.0
    for k in range(n - 1, -1, -1):
        dp = dp * x + p
        p = p * x + coeffs[k]
    return p, dp


def newton_maehly_root(coeffs, known_roots, x0=0.0, max_iter=100, tol=1e-12):
    """Newton-Maehly 求下一个根 (已知根被降阶消去).

    降阶函数:
        g(x) = f(x) / ∏_i (x - r_i)
    Newton 步:
        x_{k+1} = x_k - g(x_k) / g'(x_k)
    其中 g'(x) = [f'(x)∏(x-r_i) - f(x)∑_j∏_{i≠j}(x-r_i)] / ∏²(x-r_i)
              = (f'(x) - g(x) ∑_j 1/(x-r_j)) / 1
              = f'(x) - g(x) · ∑_j 1/(x - r_j)
    """
    x = float(x0)
    for _ in range(max_iter):
        f_val, fp_val = poly_eval_and_deriv(coeffs, x)
        if abs(fp_val) < 1e-14:
            break
        # 降阶修正
        correction = 0.0
        for r in known_roots:
            denom = x - r
            if abs(denom) < 1e-14:
                denom = 1e-14 * (1 if denom >= 0 else -1)
            correction += 1.0 / denom
        g = f_val  # 实际 g(x) = f(x)/∏(x-r_i), 但 Newton 比 = g/g' 不变
        gp = fp_val - f_val * correction
        if abs(gp) < 1e-14:
            break
        dx = -g / gp
        x += dx
        if abs(dx) < tol * (abs(x) + 1e-12):
            break
    return x


def find_path_kinks(lambdas, support_sizes):
    """检测支撑集突变点 (路径折点).

    将支撑集大小 s(λ) 视为 λ 的分段常数函数, 用 Newton-Maehly
    对 s 的多项式插值求导数的根, 定位折点.
    """
    s = np.array(support_sizes, dtype=float)
    # 有限差分近似导数
    ds = np.gradient(s, lambdas)
    # 在 ds 的符号变化处定位折点
    kinks = []
    for i in range(1, len(ds)):
        if ds[i] * ds[i - 1] < 0:
            # 线性插值
            lam_kink = 0.5 * (lambdas[i] + lambdas[i - 1])
            kinks.append(lam_kink)
    # 用 Newton-Maehly 细化 (拟合 4 阶多项式)
    if len(kinks) >= 2:
        coeffs = np.polyfit(lambdas, ds, min(4, len(lambdas) - 1))[::-1]
        refined = []
        known = []
        for k in kinks:
            r = newton_maehly_root(coeffs, known, x0=k)
            if lambdas[0] >= r >= lambdas[-1]:
                refined.append(r)
                known.append(r)
        return refined if refined else kinks
    return kinks


# ----------------------------------------------------------------------
# 交叉验证
# ----------------------------------------------------------------------
def cross_validate_lambda(A, y, lambdas, n_folds=5, seed=0):
    """K-折交叉验证选择 λ."""
    rng = np.random.RandomState(seed)
    m = len(y)
    idx = rng.permutation(m)
    folds = np.array_split(idx, n_folds)
    cv_errors = np.zeros(len(lambdas))

    for fold_idx in folds:
        mask = np.ones(m, dtype=bool)
        mask[fold_idx] = False
        A_tr, y_tr = A[mask], y[mask]
        A_val, y_val = A[~mask], y[~mask]
        x_warm = np.zeros(A.shape[1])
        for j, lam in enumerate(lambdas):
            x_sol, _ = fista_l1(A_tr, y_tr, lam, x0=x_warm,
                                max_iter=500, tol=1e-6)
            x_warm = x_sol
            cv_errors[j] += np.sum((A_val @ x_sol - y_val) ** 2) / len(y_val)
    cv_errors /= n_folds
    best_idx = int(np.argmin(cv_errors))
    return lambdas[best_idx], best_idx, cv_errors
