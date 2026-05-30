
import numpy as np


def degree(root, adj_row, adj, mask, node_num):
    deg = np.zeros(node_num, dtype=int)
    perm = np.zeros(node_num, dtype=int)
    iccsze = 0

    if node_num < 1:
        return deg, iccsze, perm

    lvlend = 0
    lnbr = 1
    perm[0] = root
    iccsze = 1
    mask[root] = 0

    while lvlend < lnbr:
        lbegin = lvlend + 1
        lvlend = lnbr
        for i in range(lbegin - 1, lvlend):
            node = perm[i]
            jstrt = adj_row[node]
            jstop = adj_row[node + 1] - 1
            for j in range(jstrt, jstop + 1):
                nbr = adj[j]
                if mask[nbr] != 0:
                    lnbr += 1
                    mask[nbr] = 0
                    perm[lnbr - 1] = nbr

    for i in range(iccsze):
        node = perm[i]
        jstrt = adj_row[node]
        jstop = adj_row[node + 1] - 1
        deg[node] = 0
        for j in range(jstrt, jstop + 1):
            if mask[adj[j]] == 0:
                deg[node] += 1
        mask[node] = 1

    return deg, iccsze, perm


def rcm(root, adj_row, adj, mask, node_num):
    if node_num < 1:
        raise ValueError("RCM: illegal NODE_NUM.")
    if root < 0 or root >= node_num:
        raise ValueError("RCM: illegal ROOT.")

    deg, iccsze, perm = degree(root, adj_row, adj, mask.copy(), node_num)
    mask[root] = 0

    if iccsze <= 1:
        return mask, perm, iccsze

    lvlend = 0
    lnbr = 1

    while lvlend < lnbr:
        lbegin = lvlend + 1
        lvlend = lnbr
        for i in range(lbegin - 1, lvlend):
            node = perm[i]
            jstrt = adj_row[node]
            jstop = adj_row[node + 1] - 1
            fnbr = lnbr + 1
            for j in range(jstrt, jstop + 1):
                nbr = adj[j]
                if mask[nbr] != 0:
                    lnbr += 1
                    mask[nbr] = 0
                    perm[lnbr - 1] = nbr

            if lnbr <= fnbr:
                continue

            k = fnbr - 1
            while k < lnbr - 1:
                l = k
                k += 1
                nbr = perm[k]
                while l >= fnbr - 1:
                    lperm = perm[l]
                    if deg[lperm] <= deg[nbr]:
                        break
                    perm[l + 1] = lperm
                    l -= 1
                perm[l + 1] = nbr

    perm[:iccsze] = perm[:iccsze][::-1]
    return mask, perm, iccsze


def r83v_mv(n, a, b, c, x_vec):
    x_vec = np.asarray(x_vec).flatten()
    y = np.zeros(n)
    y[0] = b[0] * x_vec[0]
    if n > 1:
        y[0] += c[0] * x_vec[1]
        y[n - 1] = a[n - 2] * x_vec[n - 2] + b[n - 1] * x_vec[n - 1]
    for i in range(1, n - 1):
        y[i] = a[i - 1] * x_vec[i - 1] + b[i] * x_vec[i] + c[i] * x_vec[i + 1]
    return y


def r83v_cg(n, a, b, c, ax, x0, tol=1e-12, max_iter=None):
    ax = np.asarray(ax).flatten()
    x = np.asarray(x0).flatten().copy()
    if max_iter is None:
        max_iter = n

    ap = r83v_mv(n, a, b, c, x)
    r_vec = ax - ap
    p_vec = r_vec.copy()

    for _ in range(max_iter):
        ap = r83v_mv(n, a, b, c, p_vec)
        pap = np.dot(p_vec, ap)
        if abs(pap) < np.finfo(float).eps:
            break
        pr = np.dot(p_vec, r_vec)
        alpha = pr / pap
        x += alpha * p_vec
        r_vec -= alpha * ap
        rap = np.dot(r_vec, ap)
        beta = -rap / pap
        p_vec = r_vec + beta * p_vec
        if np.linalg.norm(r_vec) < tol:
            break

    return x


def build_tridiagonal_from_fem(stiffness, mass, dt, theta=0.5):
    N = stiffness.shape[0]
    lhs = mass + theta * dt * stiffness
    rhs_mat = mass - (1.0 - theta) * dt * stiffness

    a = np.zeros(N - 1)
    b = np.zeros(N)
    c = np.zeros(N - 1)

    for i in range(N):
        b[i] = lhs[i, i]
        if i > 0:
            a[i - 1] = lhs[i, i - 1]
        if i < N - 1:
            c[i] = lhs[i, i + 1]

    return a, b, c, rhs_mat
