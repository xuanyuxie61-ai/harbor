import numpy as np


def r83_np_fa(n: int, a: np.ndarray):
    if n < 2:
        raise ValueError("r83_np_fa requires n >= 2")
    if a.shape != (3, n):
        raise ValueError(f"r83_np_fa: a must be shape (3, {n}), got {a.shape}")

    a_lu = a.copy()
    info = 0

    for i in range(n - 1):
        if a_lu[1, i] == 0.0:
            info = i + 1
            return a_lu, info
        a_lu[2, i] = a_lu[2, i] / a_lu[1, i]
        a_lu[1, i + 1] = a_lu[1, i + 1] - a_lu[2, i] * a_lu[0, i + 1]

    if a_lu[1, n - 1] == 0.0:
        info = n

    return a_lu, info


def r83_np_sl(n: int, a_lu: np.ndarray, b: np.ndarray, job: int):
    if n < 2:
        raise ValueError("r83_np_sl requires n >= 2")
    if a_lu.shape != (3, n):
        raise ValueError("r83_np_sl: a_lu shape mismatch")
    x = b.copy()

    if job == 0:

        for i in range(1, n):
            x[i] = x[i] - a_lu[2, i - 1] * x[i - 1]

        for i in range(n - 1, -1, -1):
            x[i] = x[i] / a_lu[1, i]
            if i > 0:
                x[i - 1] = x[i - 1] - a_lu[0, i] * x[i]
    else:

        for i in range(n):
            x[i] = x[i] / a_lu[1, i]
            if i < n - 1:
                x[i + 1] = x[i + 1] - a_lu[0, i + 1] * x[i]

        for i in range(n - 2, -1, -1):
            x[i] = x[i] - a_lu[2, i] * x[i + 1]

    return x


def r83p_fa(n: int, a: np.ndarray):
    if n < 3:
        raise ValueError("r83p_fa requires n >= 3")
    if a.shape != (3, n):
        raise ValueError(f"r83p_fa: a must be shape (3, {n}), got {a.shape}")

    a_lu = np.zeros((3, n), dtype=float)


    a_lu[:, : n - 1], info = r83_np_fa(n - 1, a[:, : n - 1])
    if info != 0:
        return a_lu, None, None, None, info


    a_lu[0, 0] = a[0, 0]
    a_lu[2, n - 2] = a[2, n - 2]
    a_lu[:, n - 1] = a[:, n - 1]




    work2 = None
    work3 = None
    work4 = None
    info = n
    return a_lu, work2, work3, work4, info


def r83p_sl(n: int, a_lu: np.ndarray, b: np.ndarray, job: int,
            work2: np.ndarray, work3: np.ndarray, work4: float):
    if n < 3:
        raise ValueError("r83p_sl requires n >= 3")
    x = b.copy()

    if job == 0:



        raise NotImplementedError("Hole 1: r83p_sl job==0 branch not implemented")
    else:

        x[: n - 1] = r83_np_sl(n - 1, a_lu[:, : n - 1], x[: n - 1], 1)

        x[n - 1] = x[n - 1] - a_lu[2, n - 1] * x[0] - a_lu[0, n - 1] * x[n - 2]

        x[n - 1] = x[n - 1] / work4

        x[: n - 1] = x[: n - 1] - work3 * x[n - 1]

    return x
