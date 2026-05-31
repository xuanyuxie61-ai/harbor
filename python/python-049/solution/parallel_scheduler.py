#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 parallel_scheduler.py
 
 融合种子项目：
   - 1196_task_division：任务在处理器间的分配
 
 科学功能：
   多子域并行计算的任务调度器。
   
   大规模海啸模拟需要将计算域划分为多个子域，
   分配给不同处理器并行计算。本模块使用 task_division 的
   任务分配算法，实现负载均衡的并行调度。
 
 核心数学公式：
 
   1) 任务分配算法（来源于 task_division）：
      给定 T 个任务和 P 个处理器，每个处理器分配约 T/P 个任务。
      
      处理器 p 的任务数：
        tasks_p = round(T_remaining / P_remaining)
        
      保证每个处理器获得连续的任务范围 [i_lo, i_hi]。
   
   2) 负载均衡指标：
      负载不平衡度 = (max(tasks_p) - min(tasks_p)) / (T/P)
      
      理想情况下，不平衡度应接近 0。
   
   3) 在海啸模拟中的应用：
      将二维计算域沿 x 或 y 方向划分为若干条带（strip），
      每条带分配给一个处理器。
      
      子域边界需要交换 ghost cell 数据：
        u_{ghost} = u_{neighbor}  （Dirichlet 型）
        或
        u_{ghost} = 2·u_{boundary} - u_{interior}  （反射型）
   
   4) 通信开销模型：
      总时间 = 计算时间 + 通信时间
              = max_p(T_compute,p) + α · P + β · data_size
      
      其中 α 为启动延迟，β 为带宽倒数。
"""

import numpy as np


class ParallelScheduler:
    """
    并行任务调度器。
    """
    
    def __init__(self, n_tasks, n_processors):
        """
        Parameters
        ----------
        n_tasks : int
            总任务数
        n_processors : int
            处理器数量
        """
        self.n_tasks = n_tasks
        self.n_processors = n_processors
        
        if n_tasks <= 0:
            raise ValueError("任务数必须为正")
        if n_processors <= 0:
            raise ValueError("处理器数必须为正")
    
    def divide_tasks(self):
        """
        将任务分配给处理器（来源于 task_division 核心算法）。
        
        Returns
        -------
        task_map : dict
            {processor_id: (task_start, task_end)}
        """
        task_map = {}
        
        i_hi = 0
        task_remain = self.n_tasks
        proc_remain = self.n_processors
        
        for proc in range(self.n_processors):
            # 每个处理器的任务数 = round(剩余任务数 / 剩余处理器数)
            task_proc = self._div_rounded(task_remain, proc_remain)
            
            proc_remain -= 1
            task_remain -= task_proc
            
            i_lo = i_hi + 1
            i_hi = i_hi + task_proc
            
            task_map[proc] = (i_lo, i_hi)
        
        return task_map
    
    def _div_rounded(self, a, b):
        """
        四舍五入除法。
        
        来源于 i4_div_rounded。
        """
        if b == 0:
            raise ZeroDivisionError
        
        result = a // b
        remainder = a % b
        
        # 四舍五入
        if remainder * 2 >= b:
            result += 1
        
        return result
    
    def compute_load_balance(self, task_map):
        """
        计算负载均衡指标。
        
        Returns
        -------
        imbalance : float
            负载不平衡度
        max_load : int
            最大负载
        min_load : int
            最小负载
        """
        loads = [end - start + 1 for start, end in task_map.values()]
        
        max_load = max(loads)
        min_load = min(loads)
        avg_load = self.n_tasks / self.n_processors
        
        imbalance = (max_load - min_load) / avg_load if avg_load > 0 else 0.0
        
        return imbalance, max_load, min_load
    
    def decompose_domain_1d(self, nx, ny, decomp_axis=0):
        """
        一维条带分解。
        
        将二维域沿指定轴分解为条带子域。
        
        Parameters
        ----------
        nx, ny : int
            网格尺寸
        decomp_axis : int
            分解轴（0=y方向条带，1=x方向条带）
            
        Returns
        -------
        subdomains : list
            每个子域的 (i_start, i_end, j_start, j_end)
        """
        n_cells = nx if decomp_axis == 1 else ny
        
        task_map = self.divide_tasks()
        subdomains = []
        
        for proc_id, (start, end) in task_map.items():
            if decomp_axis == 1:  # x方向条带
                i_start = start - 1
                i_end = end - 1
                j_start = 0
                j_end = ny - 1
            else:  # y方向条带
                i_start = 0
                i_end = nx - 1
                j_start = start - 1
                j_end = end - 1
            
            subdomains.append({
                'proc_id': proc_id,
                'i_start': i_start,
                'i_end': i_end,
                'j_start': j_start,
                'j_end': j_end
            })
        
        return subdomains
    
    def ghost_cell_exchange(self, local_field, subdomain, all_subdomains, ghost_width=1):
        """
        模拟 ghost cell 数据交换。
        
        在并行计算中，子域边界需要与相邻子域交换 ghost cell 数据。
        
        Parameters
        ----------
        local_field : ndarray
            本地子域数据
        subdomain : dict
            本地子域信息
        all_subdomains : list
            所有子域信息
        ghost_width : int
            ghost cell 宽度
            
        Returns
        -------
        field_with_ghost : ndarray
            带 ghost cell 的扩展场
        """
        ny_local, nx_local = local_field.shape
        
        # 创建带 ghost cell 的扩展数组
        field_ext = np.zeros((ny_local + 2 * ghost_width, nx_local + 2 * ghost_width))
        field_ext[ghost_width:ghost_width+ny_local, ghost_width:ghost_width+nx_local] = local_field
        
        # 找到相邻子域并填充 ghost cell
        for neighbor in all_subdomains:
            if neighbor['proc_id'] == subdomain['proc_id']:
                continue
            
            # 简化的 ghost cell 填充（使用零值或反射）
            # 实际并行程序中需要从邻居接收数据
            if neighbor['j_end'] == subdomain['j_start'] - 1:
                # 上方邻居
                field_ext[0:ghost_width, ghost_width:ghost_width+nx_local] = 0.0
            elif neighbor['j_start'] == subdomain['j_end'] + 1:
                # 下方邻居
                field_ext[-ghost_width:, ghost_width:ghost_width+nx_local] = 0.0
        
        return field_ext
    
    def schedule_time_step(self, subdomains, dt_per_subdomain=None):
        """
        为每个子域调度时间步。
        
        允许不同子域使用不同的时间步长（局部时间步进）。
        
        Parameters
        ----------
        subdomains : list
            子域列表
        dt_per_subdomain : list, optional
            每个子域的建议时间步
            
        Returns
        -------
        schedule : dict
            每个子域的时间步和同步策略
        """
        if dt_per_subdomain is None:
            dt_per_subdomain = [1.0] * len(subdomains)
        
        schedule = {}
        min_dt = min(dt_per_subdomain)
        
        for i, sub in enumerate(subdomains):
            # 局部时间步为最小时间步的整数倍
            ratio = max(1, round(dt_per_subdomain[i] / min_dt))
            local_dt = ratio * min_dt
            
            schedule[sub['proc_id']] = {
                'dt': local_dt,
                'sync_interval': ratio,
                'subdomain': sub
            }
        
        return schedule
