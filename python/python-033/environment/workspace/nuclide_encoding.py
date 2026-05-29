"""
nuclide_encoding.py
基于种子项目 057_atbash 和 774_monoalphabetic 的编码思想

在核天体物理中，镜像核（mirror nuclei）和电荷对称性具有重要的物理意义。
ATBASH 式的对称映射可用于描述质子-中子镜像：
    对于核素 (Z,N)，其镜像核为 (N,Z)。

单字母替换式的编码可用于构建核素标识符到计算索引的映射，
支持快速查找和核素网格遍历。

此外，引入高斯整数螺旋（种子项目 456_gaussian_prime_spiral）的思想，
将核素图 (N,Z) 映射到复整数平面，利用高斯素数路径遍历核素空间，
模拟 r 过程的核合成路径。
"""

import numpy as np


def atbash_mirror_map(nuclide_list):
    """
    对核素列表应用镜像映射（ATBASH 思想）：
    将每个核素 (Z, N) 映射为其镜像核 (N, Z)。

    参数:
        nuclide_list : list of tuple, [(Z1,N1,A1), (Z2,N2,A2), ...]

    返回:
        mirrored : list of tuple, 镜像核素列表
    """
    mirrored = []
    for z, n, a in nuclide_list:
        mirrored.append((n, z, a))
    return mirrored


def monoalphabetic_nuclide_encode(nuclide_list, key_seed=42):
    """
    使用伪随机单字母替换对核素进行编码。
    生成一个从 (Z,N) 对到唯一整数的双射映射。

    参数:
        nuclide_list : list of tuple
        key_seed : int, 随机种子

    返回:
        encode_map : dict, (Z,N) -> int
        decode_map : dict, int -> (Z,N,A)
    """
    rng = np.random.default_rng(key_seed)
    indices = np.arange(len(nuclide_list))
    rng.shuffle(indices)
    encode_map = {}
    decode_map = {}
    for i, (z, n, a) in enumerate(nuclide_list):
        code = int(indices[i])
        encode_map[(z, n)] = code
        decode_map[code] = (z, n, a)
    return encode_map, decode_map


def is_gaussian_prime(a, b):
    """
    判断高斯整数 a+bi 是否为高斯素数。

    判别准则：
    1. 若 a=0: |b| 是普通素数且 |b| ≡ 3 (mod 4)
    2. 若 b=0: |a| 是普通素数且 |a| ≡ 3 (mod 4)
    3. 若 a,b 均非零: a²+b² 是普通素数

    参数:
        a, b : int

    返回:
        bool
    """
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    if a == 0:
        return is_prime(abs(b)) and (abs(b) % 4 == 3)
    if b == 0:
        return is_prime(abs(a)) and (abs(a) % 4 == 3)
    return is_prime(a * a + b * b)


def gaussian_prime_spiral_trajectory(c_start, d_start, max_steps=1000):
    """
    生成高斯素数螺旋轨迹。
    从起点 c_start 出发，沿方向 d_start ∈ {1, i, -1, -i} 前进，
    每步 c <- c + d；若新位置 c 是高斯素数，则方向逆时针旋转 90°：d <- d * i。
    当轨迹回到起点时停止。

    在核物理中，该轨迹可用于遍历核素图 (N,Z)，其中 N 对应实部，Z 对应虚部。

    参数:
        c_start : complex, 起始位置
        d_start : complex, 初始方向 (1, 1j, -1, -1j)
        max_steps : int, 最大步数

    返回:
        trajectory : list of complex, 轨迹点
    """
    c = complex(c_start)
    d = complex(d_start)
    trajectory = [c]
    visited = {c}
    for _ in range(max_steps):
        c = c + d
        a, b = int(round(c.real)), int(round(c.imag))
        c = complex(a, b)
        if is_gaussian_prime(a, b):
            d = d * 1j  # 逆时针旋转
        if c in visited:
            break
        trajectory.append(c)
        visited.add(c)
    return trajectory


def build_nuclide_grid_path(z_min, z_max, n_min, n_max):
    """
    利用高斯素数螺旋在核素图 (Z,N) 上生成一条遍历路径，
    模拟 r 过程核素合成的可能路径。

    参数:
        z_min, z_max : int, 质子数范围
        n_min, n_max : int, 中子数范围

    返回:
        path : list of tuple, [(Z,N), ...]
    """
    center_z = (z_min + z_max) // 2
    center_n = (n_min + n_max) // 2
    c_start = complex(center_n, center_z)
    trajectory = gaussian_prime_spiral_trajectory(c_start, 1, max_steps=2000)
    path = []
    for c in trajectory:
        n, z = int(round(c.real)), int(round(c.imag))
        if z_min <= z <= z_max and n_min <= n <= n_max:
            path.append((z, n))
    # 去重并保持顺序
    seen = set()
    unique_path = []
    for p in path:
        if p not in seen:
            unique_path.append(p)
            seen.add(p)
    return unique_path


def test_nuclide_encoding():
    """自包含测试"""
    nuclides = [(26, 30, 56), (82, 126, 208), (92, 146, 238)]
    mirrored = atbash_mirror_map(nuclides)
    print(f"[nuclide_encoding] Mirror mapping: {nuclides} -> {mirrored}")

    enc, dec = monoalphabetic_nuclide_encode(nuclides, key_seed=123)
    print(f"[nuclide_encoding] Encoding map: {enc}")

    # 测试高斯素数螺旋
    traj = gaussian_prime_spiral_trajectory(0+0j, 1, max_steps=100)
    print(f"[nuclide_encoding] Gaussian prime spiral length: {len(traj)}")

    path = build_nuclide_grid_path(20, 40, 20, 50)
    print(f"[nuclide_encoding] Nuclide grid path length: {len(path)}")


if __name__ == "__main__":
    test_nuclide_encoding()
