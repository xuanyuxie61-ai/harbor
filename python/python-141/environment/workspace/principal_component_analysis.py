
import numpy as np


class PrincipalComponentAnalysis:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, A):
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("A必须为二维矩阵")
        m, n = A.shape


        self.mean_ = np.mean(A, axis=1, keepdims=True)
        A_centered = A - self.mean_


        if m > n:
            L = A_centered.T @ A_centered
            eigvals, eigvecs = np.linalg.eigh(L)

            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]


            components = A_centered @ eigvecs

            norms = np.linalg.norm(components, axis=0)
            norms[norms < 1e-12] = 1.0
            components = components / norms

            eigvals = eigvals / (n - 1) if n > 1 else eigvals
        else:
            C = (A_centered @ A_centered.T) / max(n - 1, 1)
            eigvals, components = np.linalg.eigh(C)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            components = components[:, idx]

        self.explained_variance_ = eigvals
        total_var = np.sum(eigvals)
        self.explained_variance_ratio_ = eigvals / total_var if total_var > 0 else np.zeros_like(eigvals)


        num_good = np.sum(eigvals > 1e-10)
        if self.n_components is None:
            self.n_components = num_good
        else:
            self.n_components = min(self.n_components, num_good, m)

        self.components_ = components[:, :self.n_components]
        self.explained_variance_ = eigvals[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]

    def transform(self, A):
        if self.components_ is None:
            raise RuntimeError("必须先调用fit()")
        A = np.asarray(A, dtype=np.float64)
        A_centered = A - self.mean_
        return self.components_.T @ A_centered

    def inverse_transform(self, scores):
        if self.components_ is None:
            raise RuntimeError("必须先调用fit()")
        scores = np.asarray(scores, dtype=np.float64)
        return self.mean_ + self.components_ @ scores

    def reconstruct_with_k_components(self, A, k):
        if k > self.n_components:
            raise ValueError(f"k不能超过已保留的主成分数{self.n_components}")
        A = np.asarray(A, dtype=np.float64)
        A_centered = A - self.mean_
        scores = self.components_[:, :k].T @ A_centered
        return self.mean_ + self.components_[:, :k] @ scores


def volatility_surface_pca(maturities, strikes, iv_surface, n_pcs=3):
    iv_surface = np.asarray(iv_surface, dtype=np.float64)
    if iv_surface.ndim != 2:
        raise ValueError("iv_surface必须为二维矩阵")


    m, n = iv_surface.shape


    A = iv_surface.T

    pca = PrincipalComponentAnalysis(n_components=n_pcs)
    pca.fit(A)


    pc_loadings = pca.components_


    scores = pca.transform(A)


    recon = pca.inverse_transform(scores).T

    return {
        'maturities': maturities,
        'strikes': strikes,
        'pc_loadings': pc_loadings,
        'scores': scores,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'reconstructed_surface': recon,
        'mean_curve': pca.mean_.flatten()
    }


def correlated_volatility_factors(cov_matrix, n_factors=None):
    cov = np.asarray(cov_matrix, dtype=np.float64)
    m = cov.shape[0]
    if n_factors is None:
        n_factors = m

    pca = PrincipalComponentAnalysis(n_components=n_factors)


    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]


    loadings = eigvecs[:, :n_factors] * np.sqrt(np.maximum(eigvals[:n_factors], 0.0))

    total_var = np.sum(np.maximum(eigvals, 0.0))
    explained_ratio = np.maximum(eigvals[:n_factors], 0.0) / total_var if total_var > 0 else np.zeros(n_factors)

    return {
        'loadings': loadings,
        'eigenvalues': eigvals[:n_factors],
        'explained_variance_ratio': explained_ratio,
        'cumulative_variance_ratio': np.cumsum(explained_ratio)
    }
