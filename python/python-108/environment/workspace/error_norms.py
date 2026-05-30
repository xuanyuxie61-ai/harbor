# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Optional, Tuple


class ErrorNorms:

    @staticmethod
    def l1_norm_discrete(values: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        values = np.asarray(values)
        if volumes is None:
            return float(np.sum(np.abs(values)))
        volumes = np.asarray(volumes)
        if values.shape != volumes.shape:
            raise ValueError("values 与 volumes 形状不匹配")
        return float(np.sum(np.abs(values) * volumes))

    @staticmethod
    def l2_norm_discrete(values: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        values = np.asarray(values)
        if volumes is None:
            return float(np.sqrt(np.sum(values ** 2)))
        volumes = np.asarray(volumes)
        return float(np.sqrt(np.sum((values ** 2) * volumes)))

    @staticmethod
    def linf_norm(values: np.ndarray) -> float:
        return float(np.max(np.abs(values)))

    @staticmethod
    def relative_l2_error(approx: np.ndarray, exact: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        diff = approx - exact
        num = ErrorNorms.l2_norm_discrete(diff, volumes)
        den = ErrorNorms.l2_norm_discrete(exact, volumes)
        if den < 1e-30:
            return num
        return num / den

    @staticmethod
    def relative_l1_error(approx: np.ndarray, exact: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        diff = approx - exact
        num = ErrorNorms.l1_norm_discrete(diff, volumes)
        den = ErrorNorms.l1_norm_discrete(exact, volumes)
        if den < 1e-30:
            return num
        return num / den

    @staticmethod
    def convergence_order(errors: np.ndarray, resolutions: np.ndarray) -> np.ndarray:
        errors = np.asarray(errors)
        resolutions = np.asarray(resolutions)
        if len(errors) != len(resolutions):
            raise ValueError("errors 与 resolutions 长度必须相同")
        if len(errors) < 2:
            return np.array([])
        p = np.log(errors[1:] / errors[:-1]) / np.log(resolutions[1:] / resolutions[:-1])
        return p

    @staticmethod
    def richardson_extrapolation(uh: np.ndarray, uh2: np.ndarray, p: float = 2.0) -> np.ndarray:
        factor = 2.0 ** p
        return (factor * uh2 - uh) / (factor - 1.0)

    @staticmethod
    def residual_norm(residual: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        return ErrorNorms.l2_norm_discrete(residual, volumes)

    @staticmethod
    def compute_quality_metrics(field: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
        arr = np.asarray(field)
        if mask is not None:
            arr = arr[mask.astype(bool)]
        return {
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "dynamic_range_db": 20.0 * np.log10((np.max(np.abs(arr)) + 1e-30) / (np.min(np.abs(arr)) + 1e-30)),
        }
