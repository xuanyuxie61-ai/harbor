"""
config_io.py
============

格点 QCD 规范构型的输入/输出管理.

支持:
    1. 文本格式 (用于小规模可重现实验)
    2. 二进制格式 (numpy .npy, 用于效率)
    3. 顺序命名约定 (配置链)

文件格式:
---------
文本格式:
    Header:
        # LATTICE_QCD_CONFIG
        Ns = ...
        Nt = ...
        beta = ...
        action_type = ...
        trajectory = ...
        plaquette = ...
        polyakov_loop = ...
    Body:
        对每个 link: (mu, idx, Re(U_11), Im(U_11), ..., Re(U_33), Im(U_33))

二进制格式:
    numpy.save: shape = (4, volume, 3, 3), dtype = complex128
    metadata 在 .json 文件中.

本模块融合种子项目:
  - 431_filum: 文件顺序命名与读写
  - 351_fd_to_tec: 场数据导出
  - 1136_SonyResearch_SVG_baseline: 基线配置管理
"""

import numpy as np
import os
from typing import Dict, Tuple, Optional
from lattice_geometry import LatticeGeometry
from gauge_field import GaugeField


# ============================================================
# 配置文件顺序命名
# ============================================================

def next_config_number(directory: str, prefix: str = 'cfg') -> int:
    """找到目录中的下一个可用配置编号.

    扫描文件: prefix_XXXX.npy 或 prefix_XXXX.txt
    返回最大编号 + 1.
    """
    if not os.path.exists(directory):
        return 0
    max_num = -1
    for f in os.listdir(directory):
        if f.startswith(prefix) and (f.endswith('.npy') or f.endswith('.txt')):
            try:
                num = int(f[len(prefix) + 1:f.rfind('.')])
                max_num = max(max_num, num)
            except ValueError:
                continue
    return max_num + 1


def config_filename(number: int, directory: str,
                     prefix: str = 'cfg', fmt: str = 'binary') -> str:
    """生成配置文件名."""
    ext = '.npy' if fmt == 'binary' else '.txt'
    return os.path.join(directory, f'{prefix}_{number:04d}{ext}')


def metadata_filename(number: int, directory: str,
                       prefix: str = 'cfg') -> str:
    """生成元数据文件名 (JSON)."""
    return os.path.join(directory, f'{prefix}_{number:04d}.meta.txt')


# ============================================================
# 保存配置
# ============================================================

def save_configuration(gf: GaugeField, action,
                        number: int, directory: str,
                        trajectory: int = 0,
                        prefix: str = 'cfg',
                        fmt: str = 'binary'):
    """保存规范构型及元数据.

    参数:
        gf: 规范场
        action: 规范作用量
        number: 配置编号
        directory: 输出目录
        trajectory: 当前轨迹号
        prefix: 文件名前缀
        fmt: 'binary' 或 'text'
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 保存 link 矩阵
    data_path = config_filename(number, directory, prefix, fmt)
    if fmt == 'binary':
        np.save(data_path, gf.links)
    else:
        _save_text(gf, data_path)

    # 保存元数据
    meta_path = metadata_filename(number, directory, prefix)
    plaq = gf.avg_plaquette()
    L_avg = gf.avg_polyakov_loop()
    S_g = action.total_action(gf)
    unitarity = gf.unitarity_deviation()

    with open(meta_path, 'w') as f:
        f.write(f"# LATTICE_QCD_CONFIG_METADATA\n")
        f.write(f"Ns = {gf.geom.Ns}\n")
        f.write(f"Nt = {gf.geom.Nt}\n")
        f.write(f"beta = {action.beta}\n")
        f.write(f"action_type = {action.description}\n")
        f.write(f"trajectory = {trajectory}\n")
        f.write(f"plaquette = {plaq:.12e}\n")
        f.write(f"polyakov_loop_real = {L_avg.real:.12e}\n")
        f.write(f"polyakov_loop_imag = {L_avg.imag:.12e}\n")
        f.write(f"action_value = {S_g:.12e}\n")
        f.write(f"unitarity_deviation = {unitarity:.6e}\n")


def _save_text(gf: GaugeField, filepath: str):
    """文本格式保存 (可读性优先)."""
    with open(filepath, 'w') as f:
        f.write(f"# LATTICE_QCD_CONFIG\n")
        f.write(f"Ns = {gf.geom.Ns}\n")
        f.write(f"Nt = {gf.geom.Nt}\n")
        f.write(f"volume = {gf.geom.volume}\n")
        for mu in range(4):
            for idx in range(gf.geom.volume):
                u = gf.links[mu, idx]
                values = []
                for i in range(3):
                    for j in range(3):
                        values.append(f"{u[i,j].real:.12e} {u[i,j].imag:.12e}")
                f.write(f"LINK {mu} {idx} " + " ".join(values) + "\n")


# ============================================================
# 加载配置
# ============================================================

def load_configuration(geometry: LatticeGeometry,
                        number: int, directory: str,
                        prefix: str = 'cfg',
                        fmt: str = 'binary') -> Tuple[GaugeField, Dict]:
    """加载规范构型及其元数据.

    返回:
        (gf, metadata)
    """
    data_path = config_filename(number, directory, prefix, fmt)
    gf = GaugeField(geometry, initial='cold')

    if fmt == 'binary':
        gf.links = np.load(data_path)
    else:
        _load_text(gf, data_path)

    # 加载元数据
    meta_path = metadata_filename(number, directory, prefix)
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                try:
                    metadata[key] = float(val)
                except ValueError:
                    metadata[key] = val

    return gf, metadata


def _load_text(gf: GaugeField, filepath: str):
    """文本格式加载."""
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('Ns') or line.startswith('Nt'):
                continue
            if not line.startswith('LINK'):
                continue
            parts = line.split()
            mu = int(parts[1])
            idx = int(parts[2])
            values = parts[3:]
            u = np.zeros((3, 3), dtype=np.complex128)
            k = 0
            for i in range(3):
                for j in range(3):
                    u[i, j] = float(values[k]) + 1j * float(values[k + 1])
                    k += 2
            gf.links[mu, idx] = u


# ============================================================
# 配置序列管理
# ============================================================

class ConfigurationChain:
    """管理 Markov 链上的配置序列.

    支持:
        - 按编号顺序访问配置
        - 跳过烧预期 (thermalization)
        - 测量间隔控制
    """

    def __init__(self, directory: str, prefix: str = 'cfg'):
        self.directory = directory
        self.prefix = prefix
        self._configs = self._scan_configs()

    def _scan_configs(self) -> list:
        """扫描目录中的所有配置."""
        configs = []
        if not os.path.exists(self.directory):
            return configs
        for f in sorted(os.listdir(self.directory)):
            if f.startswith(self.prefix) and f.endswith('.meta.txt'):
                try:
                    num = int(f[len(self.prefix) + 1:f.rfind('.meta')])
                    configs.append(num)
                except ValueError:
                    continue
        return configs

    @property
    def available(self) -> list:
        return self._configs

    @property
    def n_configs(self) -> int:
        return len(self._configs)

    def get_config(self, idx: int, geometry: LatticeGeometry) -> Tuple[GaugeField, Dict]:
        """按序号 (第 idx 个可用配置) 加载."""
        if idx >= len(self._configs):
            raise IndexError(f"配置 {idx} 不存在 (共 {len(self._configs)} 个)")
        num = self._configs[idx]
        return load_configuration(geometry, num, self.directory, self.prefix)
