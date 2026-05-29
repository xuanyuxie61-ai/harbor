# -*- coding: utf-8 -*-
"""
data_io.py

博士级数据输入输出与结构化解析工具

融合原项目算法：
- 224_cplex_solution_read 的 XML/结构化数据解析思想
- 1066_set_theory 的整数向量格式化输出
- 585_image_sample 的坐标向量格式化写入

科学应用场景：
结晶过程模拟产生大量的时序数据（CSD、浓度、温度、矩量等）。
本模块提供标准化的数据读写、索引管理和结果格式化功能。
"""

import numpy as np
import json


def r8vec2_write(filename, x, y, fmt="%.8e"):
    """
    将配对实数向量写入格式化文本文件。

    参数：
        filename : str
        x, y : ndarray
            配对的向量
        fmt : str
            格式字符串
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Paired vectors, n = {n}\n")
        for i in range(n):
            f.write(f"{fmt}  {fmt}\n" % (x[i], y[i]))


def i4vec_transpose_print(vec, title="", elems_per_line=10):
    """
    格式化打印整数向量（转置布局）。

    参数：
        vec : array-like
        title : str
        elems_per_line : int
    """
    vec = np.asarray(vec, dtype=int)
    n = vec.size
    if title:
        print(title)
    for ilo in range(0, n, elems_per_line):
        ihi = min(ilo + elems_per_line, n)
        line = " ".join(f"{v:6d}" for v in vec[ilo:ihi])
        print(line)


def index_set_to_string(index_set, name="I"):
    """
    将整数索引集转换为数学字符串表示。

    例如：{0, 1, 2, 5} → "I = {0, 1, 2, 5}"
    """
    indices = sorted(set(int(i) for i in index_set))
    if not indices:
        return f"{name} = ∅"
    return f"{name} = {{{', '.join(str(i) for i in indices)}}}"


def parse_solution_vector(text_data, var_prefix="x", clean_tol=1e-6):
    """
    从结构化文本中解析解向量。

    模拟 CPLEX 解读取：从文本行中提取变量名和值，
    例如 "x42 = 3.14159" → x[42] = 3.14159

    参数：
        text_data : str
            输入文本
        var_prefix : str
            变量前缀
        clean_tol : float
            清理容差（将接近整数的值四舍五入）

    返回：
        values : dict
            {index: value}
    """
    values = {}
    prefix_len = len(var_prefix)
    for line in text_data.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('=')
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        val_str = parts[1].strip()
        if not name.startswith(var_prefix):
            continue
        try:
            idx = int(name[prefix_len:])
            val = float(val_str)
            if abs(val - round(val)) < clean_tol:
                val = float(round(val))
            values[idx] = val
        except ValueError:
            continue
    return values


def write_simulation_results(filename, data_dict, metadata=None):
    """
    将模拟结果写入结构化的 JSON 格式文件。

    参数：
        filename : str
        data_dict : dict
            键值对，值为 ndarray 或标量
        metadata : dict, optional
            模拟元数据
    """
    output = {}
    if metadata:
        output['metadata'] = metadata
    output['data'] = {}
    for key, val in data_dict.items():
        if isinstance(val, np.ndarray):
            output['data'][key] = {
                'shape': list(val.shape),
                'values': val.tolist()
            }
        elif isinstance(val, (list, tuple)):
            output['data'][key] = list(val)
        else:
            output['data'][key] = float(val) if isinstance(val, (int, float, np.number)) else str(val)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def read_simulation_results(filename):
    """
    读取模拟结果文件。

    返回：
        metadata : dict
        data : dict
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = json.load(f)
    metadata = content.get('metadata', {})
    data = content.get('data', {})
    # 将列表恢复为 ndarray
    for key, val in data.items():
        if isinstance(val, dict) and 'shape' in val and 'values' in val:
            data[key] = np.array(val['values'], dtype=float).reshape(val['shape'])
    return metadata, data


def format_moment_vector(moments, names=None):
    """
    格式化矩量向量输出。

    参数：
        moments : array-like
            矩量值 [μ_0, μ_1, ..., μ_n]
        names : list of str, optional
            矩量名称

    返回：
        formatted : str
    """
    moments = np.asarray(moments, dtype=float)
    if names is None:
        names = [f"μ_{i}" for i in range(len(moments))]
    lines = []
    for name, val in zip(names, moments):
        lines.append(f"  {name:8s} = {val:14.6e}")
    return "\n".join(lines)
