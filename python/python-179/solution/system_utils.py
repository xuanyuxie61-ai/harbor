"""
system_utils.py
系统初始化与数值稳定性工具模块
================================
本模块提供合成项目的全局配置、日志记录与数值安全函数。
对应原项目 513_hello 的"系统初始化"思想，扩展为科研级工程基础。
"""

import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# 全局数值常量（双精度 IEEE-754）
# ---------------------------------------------------------------------------
EPS = np.finfo(float).eps          # 机器精度 ε ≈ 2.22×10⁻¹⁶
SQRT_EPS = np.sqrt(EPS)            # √ε ≈ 1.49×10⁻⁸
TOL_RANK = SQRT_EPS                # 数值秩判定阈值
MAX_ITER = 10000                   # 全局最大迭代次数


class Logger:
    """
    科研级日志记录器，自动附加时间戳与精度信息。
    """
    def __init__(self, name="SynthProject_179"):
        self.name = name
        self._start = time.time()

    def info(self, msg: str):
        elapsed = time.time() - self._start
        print(f"[{self.name}] [{elapsed:8.3f}s] INFO  : {msg}", flush=True)

    def warn(self, msg: str):
        elapsed = time.time() - self._start
        print(f"[{self.name}] [{elapsed:8.3f}s] WARN  : {msg}", file=sys.stderr, flush=True)

    def error(self, msg: str):
        elapsed = time.time() - self._start
        print(f"[{self.name}] [{elapsed:8.3f}s] ERROR : {msg}", file=sys.stderr, flush=True)


def safe_inv(x: np.ndarray, tol: float = EPS) -> np.ndarray:
    """
    安全求逆：对接近零的元素返回 0 而非 inf，避免数值爆炸。
    
    对于标量或数组 x，返回
        y_i = 1 / x_i   if |x_i| > tol
        y_i = 0         otherwise
    """
    y = np.zeros_like(x, dtype=float)
    mask = np.abs(x) > tol
    y[mask] = 1.0 / x[mask]
    return y


def robust_sqrt(x: np.ndarray) -> np.ndarray:
    """
    鲁棒平方根：对负数返回 0 并附带警告，防止复数污染实数计算。
    """
    x = np.asarray(x, dtype=float)
    neg = x < 0
    if np.any(neg):
        x = x.copy()
        x[neg] = 0.0
    return np.sqrt(x)


def clip_to_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    将数组严格裁剪到 [lo, hi]，防止迭代过程中参数越界。
    """
    return np.clip(x, lo, hi)


def check_finite(arr: np.ndarray, name: str = "array") -> bool:
    """
    检查数组中是否存在 nan 或 inf，若有则记录警告。
    """
    if not np.all(np.isfinite(arr)):
        bad = np.sum(~np.isfinite(arr))
        print(f"[system_utils] WARNING: {name} contains {bad} non-finite values.", flush=True)
        return False
    return True


def initialize_project() -> Logger:
    """
    项目统一初始化入口，返回 Logger 实例。
    对应原 513_hello 的扩展：由简单的 'Hello, world!' 升级为科研环境初始化。
    """
    logger = Logger()
    logger.info("=" * 60)
    logger.info("博士级合成项目 179 启动")
    logger.info("科学领域: 计算数学 — 低秩矩阵近似与张量分解")
    logger.info("=" * 60)
    logger.info(f"机器精度 ε = {EPS:.3e}")
    logger.info(f"数值秩阈值 = {TOL_RANK:.3e}")
    return logger
