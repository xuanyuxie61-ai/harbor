"""
spatial_indexing.py
三维空间填充曲线索引用于海洋分层数据

融合项目:
- 536_hilbert_curve_3d: 3D Hilbert空间填充曲线

核心科学:
海洋是三维连续介质，内波破碎事件发生在特定空间位置。
使用Hilbert空间填充曲线将三维海洋分层区域映射为一维索引，
保持空间局部性，便于高效的数据访问和并行计算。

Hilbert曲线性质:
- 连续映射: h ∈ [0, 8^r - 1] ↔ (x, y, z) ∈ [0, 2^r - 1]³
- 空间局部性: 相邻索引对应的空间点也相邻
- 递归构造: 通过八进制数字迭代构建
"""

import numpy as np


class HilbertCurve3D:
    """
    3D Hilbert空间填充曲线
    
    将三维整数坐标 (x, y, z) 与一维Hilbert索引 h 互相转换。
    递归级别为 r，每个坐标范围为 [0, 2^r - 1]。
    """
    
    def __init__(self, r):
        """
        初始化3D Hilbert曲线
        
        参数:
            r: 递归级别 (分辨率 = 2^r)
        """
        self.r = r
        self.N = 1 << r  # 2^r
        self.max_h = (1 << (3 * r)) - 1  # 8^r - 1
    
    def _transform(self, x, y, z, o, s):
        """
        基于八分象限的坐标变换
        
        参数:
            x, y, z: 坐标
            o: 八分象限索引 [0, 7]
            s: 旋转/反射状态
        """
        # 简化实现: 使用查表法处理3D Hilbert变换
        # 状态变换表 (旋转和反射)
        
        # 基本变换: 根据象限决定坐标置换和翻转
        transforms = [
            (0, 1, 2, 0),   # 象限0
            (2, 1, 0, 1),   # 象限1
            (2, 1, 0, 0),   # 象限2
            (0, 1, 2, 1),   # 象限3
            (0, 1, 2, 0),   # 象限4
            (2, 1, 0, 1),   # 象限5
            (2, 1, 0, 0),   # 象限6
            (0, 1, 2, 1),   # 象限7
        ]
        
        perm, flip = transforms[o][:3], transforms[o][3]
        coords = [x, y, z]
        new_coords = [coords[perm[i]] for i in range(3)]
        
        if flip:
            new_coords[0] = self.N - 1 - new_coords[0]
        
        return new_coords[0], new_coords[1], new_coords[2]
    
    def h_to_xyz(self, h):
        """
        Hilbert索引转换为3D坐标
        
        参数:
            h: Hilbert索引 (整数)
        
        返回:
            x, y, z: 三维坐标
        """
        h = int(h)
        x, y, z = 0, 0, 0
        
        for i in range(self.r):
            # 提取最低位的八进制数字
            o = h & 0x7
            h >>= 3
            
            # 根据象限确定坐标
            if o == 0:
                tx, ty, tz = z, x, y
                x, y, z = tx, ty, tz
                z = (1 << i) - 1 - z
            elif o == 1:
                x += (1 << i)
            elif o == 2:
                y += (1 << i)
                x, y = y, x
            elif o == 3:
                x += (1 << i)
                y += (1 << i)
            elif o == 4:
                z += (1 << i)
                x += (1 << i)
            elif o == 5:
                z += (1 << i)
                x += (1 << i)
                y, z = z, y
                y = (1 << (i + 1)) - 1 - y
            elif o == 6:
                z += (1 << i)
                y += (1 << i)
                x, z = z, x
                x = (1 << (i + 1)) - 1 - x
            elif o == 7:
                x += (1 << i)
                y += (1 << i)
                z += (1 << i)
                x, z = z, x
                z = (1 << (i + 1)) - 1 - z
        
        # 边界检查
        x = max(0, min(x, self.N - 1))
        y = max(0, min(y, self.N - 1))
        z = max(0, min(z, self.N - 1))
        
        return x, y, z
    
    def xyz_to_h(self, x, y, z):
        """
        3D坐标转换为Hilbert索引
        
        参数:
            x, y, z: 三维坐标
        
        返回:
            h: Hilbert索引
        """
        x, y, z = int(x), int(y), int(z)
        x = max(0, min(x, self.N - 1))
        y = max(0, min(y, self.N - 1))
        z = max(0, min(z, self.N - 1))
        
        h = 0
        
        for i in range(self.r - 1, -1, -1):
            h <<= 3
            mask = 1 << i
            
            # 确定当前象限
            ox = 1 if (x & mask) else 0
            oy = 1 if (y & mask) else 0
            oz = 1 if (z & mask) else 0
            
            # 简化映射
            if ox == 0 and oy == 0 and oz == 0:
                o = 0
            elif ox == 1 and oy == 0 and oz == 0:
                o = 1
            elif ox == 0 and oy == 1 and oz == 0:
                o = 2
            elif ox == 1 and oy == 1 and oz == 0:
                o = 3
            elif ox == 0 and oy == 0 and oz == 1:
                o = 4
            elif ox == 1 and oy == 0 and oz == 1:
                o = 5
            elif ox == 0 and oy == 1 and oz == 1:
                o = 6
            else:
                o = 7
            
            h |= o
        
        return h
    
    def generate_curve(self):
        """
        生成完整的Hilbert曲线点序列
        
        返回:
            points: (N³, 3) 坐标数组
        """
        n_points = self.N**3
        points = np.zeros((n_points, 3), dtype=int)
        
        for h in range(n_points):
            x, y, z = self.h_to_xyz(h)
            points[h, :] = [x, y, z]
        
        return points
    
    def locality_preservation_index(self, n_samples=1000):
        """
        计算局部性保持指数
        
        衡量一维索引距离与三维空间距离的相关系数。
        越接近1表示局部性保持越好。
        
        参数:
            n_samples: 采样对数
        
        返回:
            lpi: 局部性保持指数
        """
        n_points = self.N**3
        
        if n_samples > n_points * (n_points - 1) // 2:
            n_samples = min(n_samples, 1000)
        
        h1 = np.random.randint(0, n_points, n_samples)
        h2 = np.random.randint(0, n_points, n_samples)
        
        d_h = np.abs(h1 - h2).astype(float)
        d_xyz = np.zeros(n_samples)
        
        for i in range(n_samples):
            x1, y1, z1 = self.h_to_xyz(h1[i])
            x2, y2, z2 = self.h_to_xyz(h2[i])
            d_xyz[i] = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        
        # 相关系数
        if np.std(d_h) > 1.0e-12 and np.std(d_xyz) > 1.0e-12:
            lpi = np.corrcoef(d_h, d_xyz)[0, 1]
        else:
            lpi = 0.0
        
        return lpi


def ocean_volume_indexing(depth_levels, lat_levels, lon_levels, r=3):
    """
    将海洋三维体数据映射为Hilbert索引
    
    参数:
        depth_levels: 深度层数
        lat_levels: 纬度层数
        lon_levels: 经度层数
        r: Hilbert曲线分辨率
    
    返回:
        hc: HilbertCurve3D对象
        depth_scale: 深度缩放因子
        lat_scale: 纬度缩放因子
        lon_scale: 经度缩放因子
    """
    N = 1 << r
    
    # 缩放因子: 将各维度映射到 [0, N-1]
    depth_scale = (N - 1) / max(depth_levels - 1, 1)
    lat_scale = (N - 1) / max(lat_levels - 1, 1)
    lon_scale = (N - 1) / max(lon_levels - 1, 1)
    
    hc = HilbertCurve3D(r)
    
    return hc, depth_scale, lat_scale, lon_scale
