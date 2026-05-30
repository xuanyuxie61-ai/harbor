#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class ParallelScheduler:
    
    def __init__(self, n_tasks, n_processors):
        self.n_tasks = n_tasks
        self.n_processors = n_processors
        
        if n_tasks <= 0:
            raise ValueError("任务数必须为正")
        if n_processors <= 0:
            raise ValueError("处理器数必须为正")
    
    def divide_tasks(self):
        task_map = {}
        
        i_hi = 0
        task_remain = self.n_tasks
        proc_remain = self.n_processors
        
        for proc in range(self.n_processors):

            task_proc = self._div_rounded(task_remain, proc_remain)
            
            proc_remain -= 1
            task_remain -= task_proc
            
            i_lo = i_hi + 1
            i_hi = i_hi + task_proc
            
            task_map[proc] = (i_lo, i_hi)
        
        return task_map
    
    def _div_rounded(self, a, b):
        if b == 0:
            raise ZeroDivisionError
        
        result = a // b
        remainder = a % b
        

        if remainder * 2 >= b:
            result += 1
        
        return result
    
    def compute_load_balance(self, task_map):
        loads = [end - start + 1 for start, end in task_map.values()]
        
        max_load = max(loads)
        min_load = min(loads)
        avg_load = self.n_tasks / self.n_processors
        
        imbalance = (max_load - min_load) / avg_load if avg_load > 0 else 0.0
        
        return imbalance, max_load, min_load
    
    def decompose_domain_1d(self, nx, ny, decomp_axis=0):
        n_cells = nx if decomp_axis == 1 else ny
        
        task_map = self.divide_tasks()
        subdomains = []
        
        for proc_id, (start, end) in task_map.items():
            if decomp_axis == 1:
                i_start = start - 1
                i_end = end - 1
                j_start = 0
                j_end = ny - 1
            else:
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
        ny_local, nx_local = local_field.shape
        

        field_ext = np.zeros((ny_local + 2 * ghost_width, nx_local + 2 * ghost_width))
        field_ext[ghost_width:ghost_width+ny_local, ghost_width:ghost_width+nx_local] = local_field
        

        for neighbor in all_subdomains:
            if neighbor['proc_id'] == subdomain['proc_id']:
                continue
            


            if neighbor['j_end'] == subdomain['j_start'] - 1:

                field_ext[0:ghost_width, ghost_width:ghost_width+nx_local] = 0.0
            elif neighbor['j_start'] == subdomain['j_end'] + 1:

                field_ext[-ghost_width:, ghost_width:ghost_width+nx_local] = 0.0
        
        return field_ext
    
    def schedule_time_step(self, subdomains, dt_per_subdomain=None):
        if dt_per_subdomain is None:
            dt_per_subdomain = [1.0] * len(subdomains)
        
        schedule = {}
        min_dt = min(dt_per_subdomain)
        
        for i, sub in enumerate(subdomains):

            ratio = max(1, round(dt_per_subdomain[i] / min_dt))
            local_dt = ratio * min_dt
            
            schedule[sub['proc_id']] = {
                'dt': local_dt,
                'sync_interval': ratio,
                'subdomain': sub
            }
        
        return schedule
