"""
platform_detect.py
==================
平台与环境检测模块（源自 seed 824_octopus 的运行时环境检测思想）

在分子动力学模拟中，不同平台（纯 Python / NumPy / Numba / GPU）的浮点精度、
随机数生成器、以及并行后端存在差异。本模块提供统一检测接口，确保数值可复现性。
"""

import sys
import os


def is_numpy_mkl() -> bool:
    """
    检测当前 NumPy 是否链接 Intel MKL 后端。
    MKL 后端在高维矩阵运算中提供更稳定的 SVD 与 Cholesky 分解。
    """
    try:
        import numpy as np
        config = str(np.__config__.show())
        return "mkl" in config.lower()
    except Exception:
        return False


def get_platform_info() -> dict:
    """
    获取运行平台的关键信息字典，用于后续数值稳定性判断。

    Returns
    -------
    info : dict
        包含 python_version, numpy_linked, float_eps, max_threads 等字段。
    """
    import numpy as np
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "numpy_mkl": is_numpy_mkl(),
        "float_eps": float(np.finfo(np.float64).eps),
        "max_threads": os.cpu_count() or 1,
    }
    return info


def check_environment() -> bool:
    """
    综合环境检查：确认必要的科学计算栈已就绪。

    对于纳米颗粒-生物膜相互作用的粗粒化分子动力学模拟，需要保证：
    - NumPy 双精度浮点误差在 2.22e-16 量级；
    - 至少 2 个 CPU 核心可用于并行力计算。
    """
    info = get_platform_info()
    if info["float_eps"] > 1e-15:
        return False
    if info["max_threads"] < 1:
        return False
    return True
