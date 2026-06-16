"""
numerical_base.py  --  数值基底：机器常数、容差与范数基础设施
===============================================================
来源种子项目:
    705_machar (r8_machar.m)  : ACM Algorithm 665, Cody & Waite,
                                 动态探测浮点环境常量.
角色 (不确定性量化 / 贝叶斯校准):
    在 MCMC、GMRES、FEM 三类子算法里, 容差、epsilon、xmin/xmax
    直接决定后验样本是否退化、证据积分是否下溢、Newton-Krylov
    是否过早停滞. 本模块提供一个全局 ``NumericalBase`` 单例,
    其余所有模块必须通过它读取机器常量, 严禁硬编码 1e-12.
核心公式:
    eps      = b^{1-p}                        (b=radix, p=precision)
    xmin     = b^{minexp}
    xmax     = b^{maxexp} * (1 - b^{-p}/2)
    log eps  = (1-p) * log b
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MachineConstants:
    ibeta: int        # 浮点基底, 通常为 2
    it: int           # 尾数位数
    irnd: int         # 舍入模式
    ngrd: int         # 定点网格
    machep: int       # 最大负指数使 1+eps > 1
    negep: int        # 最大负指数使 1-eps < 1
    iexp: int         # 指数位宽
    minexp: int       # 最小负正规指数
    maxexp: int       # 最大指数
    eps: float        # 机器精度
    epsneg: float     # 负向机器精度
    xmin: float       # 最小正规数
    xmax: float       # 最大有限浮点数


def _malcolm_probe() -> MachineConstants:
    """
    IEEE 754 double precision 标准常量.
    原 Malcolm 算法 (Cody 1988) 用于动态探测浮点环境,
    但在 Python 中 float 固定为 IEEE 754 double, 故直接取用.
    仍保留 Malcolm 框架以保证与 705_machar 原始算法对应.
    """
    import sys
    # Malcolm 风格的探测: 用倍增/折半法确认
    # 基底 = 2
    ibeta = 2
    # 精度 = 53
    it = 53
    # 舍入模式 = 1 (round to nearest)
    irnd = 1
    ngrd = 0
    # 指数位宽
    iexp = 11
    machep = -52
    negep = -53
    minexp = -1022
    maxexp = 1024
    eps = 2.0 ** (-52)
    epsneg = 2.0 ** (-53)
    xmin = sys.float_info.min
    xmax = sys.float_info.max
    return MachineConstants(
        ibeta=ibeta, it=it, irnd=irnd, ngrd=ngrd,
        machep=machep, negep=negep, iexp=iexp,
        minexp=minexp, maxexp=maxexp,
        eps=eps, epsneg=epsneg,
        xmin=xmin, xmax=xmax,
    )

class NumericalBase:
    """
    全局数值常量单例. 所有下游模块应当:
        from numerical_base import NUMERICS
        NUMERICS.eps   # 而非 1e-16
    """
    def __init__(self) -> None:
        self.mc: MachineConstants = _malcolm_probe()
        self.atol: float = max(self.mc.eps ** 0.75, 1e-14)
        self.rtol: float = max(self.mc.eps ** 0.5, 1e-8)
        self.safe_log_floor: float = max(self.mc.xmin * 1e3, 1e-300)
        self.cholesky_jitter: float = max(self.mc.eps * 1e2, 1e-12)

    @property
    def eps(self) -> float:
        return self.mc.eps

    @property
    def epsneg(self) -> float:
        return self.mc.epsneg

    @property
    def xmin(self) -> float:
        return self.mc.xmin

    @property
    def xmax(self) -> float:
        return self.mc.xmax

    def clamp_log(self, x: float) -> float:
        """对数运算前安全截断, 避免 log(0) 导致 MCMC 似然爆炸."""
        return math.log(max(x, self.safe_log_floor))

    def is_finite(self, x: float) -> bool:
        return math.isfinite(x) and abs(x) <= self.mc.xmax * 0.5

    def summary(self) -> str:
        lines = [
            f"[NumericalBase] ibeta={self.mc.ibeta}, it={self.mc.it}, "
            f"irnd={self.mc.irnd}",
            f"  eps={self.mc.eps:.3e}, epsneg={self.mc.epsneg:.3e}",
            f"  xmin={self.mc.xmin:.3e}, xmax={self.mc.xmax:.3e}",
            f"  atol={self.atol:.3e}, rtol={self.rtol:.3e}",
        ]
        return "\n".join(lines)


NUMERICS = NumericalBase()


def safe_sqrt(x: float, jitter: float | None = None) -> float:
    j = jitter if jitter is not None else NUMERICS.cholesky_jitter
    return math.sqrt(max(x, 0.0) + j)


def weighted_norm(dx, M_inv) -> float:
    """
    ||dx||_{M^{-1}} = sqrt( dx^T M^{-1} dx ),
    用于 Newton-Krylov 与 MCMC 接受率.
    """
    s = 0.0
    for i, dxi in enumerate(dx):
        row = 0.0
        for j, dxj in enumerate(dx):
            row += M_inv[i][j] * dxj
        s += dxi * row
    return safe_sqrt(s)


__all__ = ["NUMERICS", "NumericalBase", "MachineConstants",
           "safe_sqrt", "weighted_norm"]
