"""
mode_classifier.py
==================
振荡模式的符号分类与量子数映射.

融合种子项目:
  - monoalphabetic (774): 单表替换加密/解密 → 模式标识符映射
  - boundary_word_drafter (106): 边界字编码/奇偶校验 → 模式特征编码

科学背景
--------
恒星振荡模式由三个量子数 (n, l, m) 标识:
  - n: 径向阶数 (径向导数数量)
  - l: 角量子数 (球谐函数阶数)
  - m: 方位角量子数 (m = -l, ..., +l)

模式命名约定:
  l = 0: 径向模 (p 模序列)
  l = 1: 偶极模 (dipole)
  l = 2: 四极模 (quadrupole)
  l = 3: 八极模 (octupole)

频率渐近关系 (Tassoul 1980):
  ν_{n,l} ≈ (n + l/2 + ε) Δν - δ₀ₗ

模式分类问题:
  给定一组观测频率, 识别每个频率对应的 (n, l, m).
  这是一个组合优化问题, 类似于密码破译 (monoalphabetic cipher).

  编码映射:
    理论量子数 → 观测标识符 (加密)
    观测标识符 → 理论量子数 (解密)

  单表替换密码的类比 (from 774):
    CODE[i] = 第 i 个字母的替换
    encrypt: plain[i] → CODE[plain[i]]
    decrypt: crypt[i] → 找到 j 使得 CODE[j] = crypt[i]

边界字编码 (from 106):
  模式特征函数的拓扑特征可以编码为边界字:
  - 每个字符代表特征函数在一个区域的行为
  - 奇偶性 (parity) 反映模式的对称性

本模块实现:
  1. 量子数到符号标识符的映射
  2. 模式频率的自动分类
  3. 特征函数拓扑编码
  4. 模式奇偶性分析
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


class ModeIdentifierMap:
    """
    模式标识符映射 (单表替换密码).

    融合 monoalphabetic (774) 项目:
    将理论量子数 (n, l, m) 映射为观测标识符, 反之亦然.

    标准编码 (26 个字母):
    A-Z 对应不同的 (n, l) 组合

    Parameters
    ----------
    encoding : str, optional
        编码表 (26 个字符的排列)
    """

    def __init__(self, encoding: Optional[str] = None):
        if encoding is None:
            # 标准编码: 按频率排序分配字母
            self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            self.code = self._generate_standard_code()
        else:
            if len(encoding) != 26:
                raise ValueError(f"编码表长度必须为 26, 收到 {len(encoding)}")
            self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            self.code = encoding

        # 构建反向映射 (解密表)
        self.decrypt_map = {}
        for i in range(26):
            self.decrypt_map[self.code[i]] = self.alphabet[i]

    def _generate_standard_code(self) -> str:
        """
        生成标准编码 (按物理优先级排序).

        排序优先级:
        1. 大频率间距 Δν 分组
        2. 角量子数 l 递增
        3. 径向阶数 n 递增

        Returns
        -------
        code : str
            26 个字符的排列
        """
        # 标准: 字母表顺序 (简化)
        # 实际中根据频率排序
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def encrypt(self, plain: str) -> str:
        """
        加密: 理论标识符 → 观测标识符.

        Parameters
        ----------
        plain : str
            理论标识符

        Returns
        -------
        crypt : str
            加密后的标识符
        """
        crypt = list(plain.upper())
        for i in range(len(crypt)):
            if 'A' <= crypt[i] <= 'Z':
                idx = ord(crypt[i]) - ord('A')
                crypt[i] = self.code[idx]
        return "".join(crypt)

    def decrypt(self, crypt: str) -> str:
        """
        解密: 观测标识符 → 理论标识符.

        Parameters
        ----------
        crypt : str
            观测标识符

        Returns
        -------
        plain : str
            解密后的标识符
        """
        plain = list(crypt.upper())
        for i in range(len(plain)):
            if plain[i] in self.decrypt_map:
                plain[i] = self.decrypt_map[plain[i]]
        return "".join(plain)

    def assign_quantum_numbers(
        self,
        frequencies: np.ndarray,
        delta_nu: float,
        epsilon: float = 0.5,
        l_max: int = 3,
    ) -> List[Dict[str, int]]:
        """
        从频率序列分配量子数.

        使用渐近关系:
          ν_{n,l} ≈ (n + l/2 + ε) Δν

        对每个频率, 找到最近的 (n, l) 组合:
          n + l/2 ≈ ν/Δν - ε

        Parameters
        ----------
        frequencies : ndarray
            排序后的频率
        delta_nu : float
            大频率间距
        epsilon : float
            渐近修正
        l_max : int
            最大角量子数

        Returns
        -------
        assignments : list of dict
            每个模式的 (n, l) 赋值
        """
        assignments = []

        for freq in frequencies:
            best_n, best_l, best_dist = 0, 0, np.inf

            for l in range(l_max + 1):
                # 从渐近关系推断 n
                n_approx = freq / delta_nu - l / 2.0 - epsilon
                n_floor = max(0, int(np.round(n_approx)))

                # 检查附近的 n 值
                for n_try in range(max(0, n_floor - 1), n_floor + 2):
                    nu_theory = (n_try + l / 2.0 + epsilon) * delta_nu
                    dist = abs(freq - nu_theory)

                    if dist < best_dist:
                        best_dist = dist
                        best_n = n_try
                        best_l = l

            # 转换为字母标识
            idx = best_n * (l_max + 1) + best_l
            letter = chr(ord('A') + (idx % 26))

            assignments.append({
                "frequency": freq,
                "n": best_n,
                "l": best_l,
                "identifier": letter,
                "residual": best_dist,
            })

        return assignments


class EigenfunctionTopologyEncoder:
    """
    特征函数拓扑编码器.

    融合 boundary_word_drafter (106) 项目:
    将特征函数的拓扑特征编码为边界字 (boundary word).

    边界字描述:
    - 每个字符代表特征函数在一个角度扇区的行为
    - 方向编码: A=东, B=东北, C=北, ... (12 方向)
    - 步长编码: 大写 = 长步, 小写 = 短步

    在恒星振荡中:
    - 径向节点数 = n
    - 角节点数 = l
    - 特征函数的符号变化编码为边界字序列

    奇偶性 (from 106):
    parity = Σ(正三角形) - Σ(负三角形)

    Parameters
    ----------
    n_angular : int
        角度离散化点数
    """

    # 12 方向编码 (融合 106 项目)
    DIRECTIONS = {
        'A': (0.5, 0.0),              # 东
        'B': (np.sqrt(3)/3, 30),       # 东北
        'C': (0.5, 60.0),              # 北偏东
        'D': (np.sqrt(3)/3, 90.0),     # 北
        'E': (0.5, 120.0),             # 北偏西
        'F': (np.sqrt(3)/3, 150.0),    # 西北
        'G': (0.5, 180.0),             # 西
        'H': (np.sqrt(3)/3, 210.0),    # 西南
        'I': (0.5, 240.0),             # 南偏西
        'J': (np.sqrt(3)/3, 270.0),    # 南
        'K': (0.5, 300.0),             # 南偏东
        'L': (np.sqrt(3)/3, 330.0),    # 东南
    }

    def __init__(self, n_angular: int = 36):
        self.n_angular = n_angular

    def encode_eigenfunction(
        self,
        xi_r: np.ndarray,
        xi_h: np.ndarray,
        l: int,
    ) -> str:
        """
        将本征函数编码为边界字.

        方法:
        1. 计算本征函数的角度分布
        2. 在每个角度扇区判断符号和梯度方向
        3. 映射为边界字符号

        Parameters
        ----------
        xi_r : ndarray
            径向位移本征函数
        xi_h : ndarray
            水平位移本征函数
        l : int
            角量子数

        Returns
        -------
        boundary_word : str
            边界字编码
        """
        n_r = len(xi_r)

        # 计算本征函数的符号变化
        sign_changes_r = np.sum(np.abs(np.diff(np.sign(xi_r))) > 0)
        sign_changes_h = np.sum(np.abs(np.diff(np.sign(xi_h))) > 0)

        # 映射为方向字母
        word = []
        total_sign_changes = int(sign_changes_r) + int(sign_changes_h)

        for i in range(min(total_sign_changes + l, 26)):
            # 根据节点数和角度量子数选择方向
            direction_idx = (i * 3 + l * 2) % 12
            direction_char = chr(ord('A') + direction_idx)

            # 大小写取决于振幅
            amp = np.abs(xi_r[min(i * n_r // max(total_sign_changes + 1, 1), n_r - 1)])
            if amp > np.mean(np.abs(xi_r)):
                word.append(direction_char.upper())
            else:
                word.append(direction_char.lower())

        return "".join(word) if word else "A"

    def compute_parity(self, boundary_word: str) -> int:
        """
        计算边界字的奇偶性.

        融合 106 项目的 word_parity:
        parity = Σ(正三角形) - Σ(负三角形)

        Parameters
        ----------
        boundary_word : str
            边界字

        Returns
        -------
        parity : int
            +1 (偶), -1 (奇), 0 (混合)
        """
        if not boundary_word:
            return 0

        # 统计大小写字母
        upper_count = sum(1 for c in boundary_word if c.isupper())
        lower_count = sum(1 for c in boundary_word if c.islower())

        if upper_count > lower_count:
            return 1  # 偶宇称
        elif lower_count > upper_count:
            return -1  # 奇宇称
        else:
            return 0  # 混合

    def mode_symmetry_check(
        self,
        xi_r: np.ndarray,
        l: int,
    ) -> Dict[str, any]:
        """
        检查模式的对称性.

        球谐函数 Y_l^m 的宇称 = (-1)^l

        Parameters
        ----------
        xi_r : ndarray
            径向本征函数
        l : int
            角量子数

        Returns
        -------
        result : dict
            expected_parity: 理论宇称
            computed_parity: 计算宇称
            is_consistent: 是否一致
        """
        expected = (-1)**l
        boundary_word = self.encode_eigenfunction(xi_r, np.zeros_like(xi_r), l)
        computed = self.compute_parity(boundary_word)

        return {
            "expected_parity": expected,
            "computed_parity": computed,
            "boundary_word": boundary_word,
            "is_consistent": (computed == expected) or (computed == 0),
            "l": l,
        }


class ModePatternRecognizer:
    """
    模式图样识别器.

    在 echelle 图中识别 p 模和 g 模的规律性图样.

    echelle 图:
    - X 轴: ν mod Δν (折叠频率)
    - Y 轴: ν / Δν (序数)
    - 相同 l 的模式形成近似垂直的 ridge

    识别算法:
    1. 计算 echelle 坐标
    2. 按 l 值分组
    3. 拟合 ridge 曲线

    Parameters
    ----------
    delta_nu : float
        大频率间距
    """

    def __init__(self, delta_nu: float):
        if delta_nu <= 0:
            raise ValueError(f"delta_nu={delta_nu} 必须为正")
        self.delta_nu = delta_nu

    def compute_echelle_coordinates(
        self,
        frequencies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 echelle 图坐标.

        x = ν mod Δν
        y = floor(ν / Δν)

        Parameters
        ----------
        frequencies : ndarray
            振荡频率

        Returns
        -------
        x_echelle : ndarray
            折叠频率
        y_echelle : ndarray
            序数
        """
        x = np.mod(frequencies, self.delta_nu)
        y = np.floor(frequencies / self.delta_nu)
        return x, y

    def identify_ridges(
        self,
        frequencies: np.ndarray,
        assignments: List[Dict],
    ) -> Dict[int, np.ndarray]:
        """
        识别 echelle 图中的 ridge.

        Parameters
        ----------
        frequencies : ndarray
            频率
        assignments : list
            量子数赋值

        Returns
        -------
        ridges : dict
            l → 频率数组
        """
        ridges = {}
        for assign in assignments:
            l = assign["l"]
            if l not in ridges:
                ridges[l] = []
            ridges[l].append(assign["frequency"])

        return {l: np.array(freqs) for l, freqs in ridges.items()}

    def fit_ridge_curvature(
        self,
        ridge_freqs: np.ndarray,
        max_curvature: float = 0.1,
    ) -> Dict[str, float]:
        """
        拟合 ridge 曲率 (二阶修正).

        渐近关系的高阶修正:
          ν_{n,l} = (n + l/2 + ε) Δν + α (n + l/2 + ε)²

        曲率 α 反映恒星内部的声学剖面的偏离.

        Parameters
        ----------
        ridge_freqs : ndarray
            ridge 上的频率
        max_curvature : float
            最大允许曲率

        Returns
        -------
        result : dict
            delta_nu: 拟合的大间距
            epsilon: 渐近修正
            curvature: 曲率
            residual_rms: 残差 RMS
        """
        if len(ridge_freqs) < 3:
            return {
                "delta_nu": self.delta_nu,
                "epsilon": 0.5,
                "curvature": 0.0,
                "residual_rms": 0.0,
            }

        n = np.arange(len(ridge_freqs))

        # 二次拟合: ν = a n² + b n + c
        coeffs = np.polyfit(n, ridge_freqs, 2)
        a, b, c = coeffs

        # 提取参数
        delta_nu_fit = b  # 线性项
        epsilon_fit = c / delta_nu_fit if delta_nu_fit > 0 else 0.5
        curvature = a / delta_nu_fit if delta_nu_fit > 0 else 0.0

        # 限制曲率范围
        curvature = np.clip(curvature, -max_curvature, max_curvature)

        # 残差
        nu_fit = a * n**2 + b * n + c
        residual_rms = np.sqrt(np.mean((ridge_freqs - nu_fit)**2))

        return {
            "delta_nu": delta_nu_fit,
            "epsilon": epsilon_fit,
            "curvature": curvature,
            "residual_rms": residual_rms,
        }
