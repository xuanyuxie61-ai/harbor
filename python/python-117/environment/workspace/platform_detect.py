
import sys
import os


def is_numpy_mkl() -> bool:
    try:
        import numpy as np
        config = str(np.__config__.show())
        return "mkl" in config.lower()
    except Exception:
        return False


def get_platform_info() -> dict:
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
    info = get_platform_info()
    if info["float_eps"] > 1e-15:
        return False
    if info["max_threads"] < 1:
        return False
    return True
