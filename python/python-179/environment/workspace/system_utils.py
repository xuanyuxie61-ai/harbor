
import sys
import time
import numpy as np




EPS = np.finfo(float).eps
SQRT_EPS = np.sqrt(EPS)
TOL_RANK = SQRT_EPS
MAX_ITER = 10000


class Logger:
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
    y = np.zeros_like(x, dtype=float)
    mask = np.abs(x) > tol
    y[mask] = 1.0 / x[mask]
    return y


def robust_sqrt(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    neg = x < 0
    if np.any(neg):
        x = x.copy()
        x[neg] = 0.0
    return np.sqrt(x)


def clip_to_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def check_finite(arr: np.ndarray, name: str = "array") -> bool:
    if not np.all(np.isfinite(arr)):
        bad = np.sum(~np.isfinite(arr))
        print(f"[system_utils] WARNING: {name} contains {bad} non-finite values.", flush=True)
        return False
    return True


def initialize_project() -> Logger:
    logger = Logger()
    logger.info("=" * 60)
    logger.info("博士级合成项目 179 启动")
    logger.info("科学领域: 计算数学 — 低秩矩阵近似与张量分解")
    logger.info("=" * 60)
    logger.info(f"机器精度 ε = {EPS:.3e}")
    logger.info(f"数值秩阈值 = {TOL_RANK:.3e}")
    return logger
