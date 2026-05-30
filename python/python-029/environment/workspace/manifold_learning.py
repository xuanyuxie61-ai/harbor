
import numpy as np


def euclidean_distance_matrix(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i, :] - X[j, :])
            D[i, j] = d
            D[j, i] = d
    return D


def sammon_mapping(X, n_components=2, max_iter=300, alpha=0.3, tol=1e-9):
    n = X.shape[0]
    D = euclidean_distance_matrix(X)

    D = np.where(D < 1e-12, 1e-12, D)


    Y = np.random.randn(n, n_components) * 0.01

    if n_features := X.shape[1] >= n_components:
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        for d in range(n_components):
            if d < len(idx):
                Y[:, d] = (X - X.mean(axis=0)) @ eigvecs[:, idx[d]]


    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-12)

    stress_history = []
    denom = np.sum(D[D > 0])

    for iteration in range(max_iter):
        D_star = euclidean_distance_matrix(Y)
        D_star = np.where(D_star < 1e-12, 1e-12, D_star)


        mask = D > 0
        stress = np.sum(((D[mask] - D_star[mask]) ** 2) / D[mask]) / denom
        stress_history.append(stress)

        if iteration > 10 and abs(stress_history[-1] - stress_history[-2]) < tol:
            break


        for i in range(n):
            grad = np.zeros(n_components)
            for j in range(n):
                if i == j:
                    continue
                diff = Y[i, :] - Y[j, :]
                factor = -2.0 * (D[i, j] - D_star[i, j]) / (D[i, j] * D_star[i, j])
                grad += factor * diff
            Y[i, :] -= alpha * grad

    return Y, stress_history


def nuclear_mass_manifold(Z_range, N_range):
    from nuclear_data_io import Nuclide

    data = []
    labels = []
    for Z in Z_range:
        for N in N_range:
            A = Z + N
            if A < Z or A < 1:
                continue
            nuc = Nuclide(Z, A)
            BE_per_A = nuc.binding_energy() / A
            Sn = nuc.neutron_separation_energy()
            Sp = nuc.proton_separation_energy()

            features = [float(N), float(Z), BE_per_A, Sn, Sp]
            data.append(features)
            labels.append(f"{A}{nuc._element_symbol() if hasattr(nuc, '_element_symbol') else ''}")

    return np.array(data), labels


def local_linear_embedding(X, n_neighbors=5, n_components=2):
    n = X.shape[0]
    D = euclidean_distance_matrix(X)


    neighbors = np.argsort(D, axis=1)[:, 1:n_neighbors + 1]


    W = np.zeros((n, n))
    for i in range(n):
        Xi = X[neighbors[i], :] - X[i, :]
        G = Xi @ Xi.T

        G += 1e-3 * np.eye(n_neighbors)
        w = np.linalg.solve(G, np.ones(n_neighbors))
        w = w / np.sum(w)
        W[i, neighbors[i]] = w


    I = np.eye(n)
    M = (I - W).T @ (I - W)


    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)

    Y = eigvecs[:, idx[1:n_components + 1]]
    return Y


def cluster_nuclides_by_stability(X, n_clusters=3):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_


def magic_number_detection(Z_list, N_list, BE_list):
    Z_list = np.array(Z_list)
    N_list = np.array(N_list)
    BE_list = np.array(BE_list)

    magic_numbers = []
    for Z in np.unique(Z_list):
        mask = Z_list == Z
        Ns = N_list[mask]
        BEs = BE_list[mask]
        idx = np.argsort(Ns)
        Ns = Ns[idx]
        BEs = BEs[idx]
        if len(Ns) < 3:
            continue
        delta2 = BEs[2:] - 2 * BEs[1:-1] + BEs[:-2]

        for i, d2 in enumerate(delta2):
            if d2 > 2.0:
                magic_numbers.append((Z, int(Ns[i + 1])))

    return magic_numbers


def binding_energy_gradient_flow(Z_range, N_range):
    from nuclear_data_io import Nuclide

    Z_vals = list(Z_range)
    N_vals = list(N_range)
    Z_grid, N_grid = np.meshgrid(Z_vals, N_vals)
    BE_grid = np.zeros_like(Z_grid, dtype=float)

    for i, N in enumerate(N_vals):
        for j, Z in enumerate(Z_vals):
            A = N + Z
            if A >= Z and A > 0:
                nuc = Nuclide(Z, A)
                BE_grid[i, j] = nuc.binding_energy() / A
            else:
                BE_grid[i, j] = np.nan


    dBE_dZ, dBE_dN = np.gradient(BE_grid)
    return Z_grid, N_grid, dBE_dZ, dBE_dN


if __name__ == "__main__":


    t = np.linspace(0, 4 * np.pi, 100)
    X_test = np.column_stack([np.cos(t), np.sin(t), t])

    Y_sammon, stress = sammon_mapping(X_test, n_components=2, max_iter=200)
    print(f"Sammon 最终应力: {stress[-1]:.6f}")

    Y_lle = local_linear_embedding(X_test, n_neighbors=5, n_components=2)
    print(f"LLE 嵌入形状: {Y_lle.shape}")


    X_nuc, labels = nuclear_mass_manifold(range(20, 30), range(20, 40))
    print(f"核素数据形状: {X_nuc.shape}")


    magic = magic_number_detection(X_nuc[:, 1].astype(int), X_nuc[:, 0].astype(int), X_nuc[:, 2] * (X_nuc[:, 0] + X_nuc[:, 1]))
    print(f"检测到的幻数 (Z, N): {magic[:10]}")
