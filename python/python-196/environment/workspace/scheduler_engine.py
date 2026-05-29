"""
scheduler_engine.py
异构任务调度引擎

包含：
- 贪心分区负载均衡（源自 partition_greedy）
- 整数规划任务映射（源自 pariomino_tiling_solver）
- 博弈论抢占策略（源自 reversi_game 的贪心最优策略）
- 多目标优化：makespan, 能耗, 可靠性

科学背景：
异构调度问题可形式化为一个混合整数非线性规划：

    决策变量:
        x_{ij} ∈ {0,1}: 任务i是否分配给处理器j
        s_i ≥ 0: 任务i的开始时间

    目标（最小化加权组合）:
        min  w1 * C_max + w2 * E_total + w3 * (1 - R_total)

    约束:
        (1) sum_j x_{ij} = 1,          forall i      (每个任务恰好分配到一个处理器)
        (2) s_i + t_{ij} * x_{ij} ≤ C_max, forall i,j  (makespan)
        (3) s_j ≥ s_i + t_{ij} + comm_{jk},  forall 依赖边 (i->j), proc(i)=k, proc(j)=l
        (4) T_j ≤ T_max_safe,          forall j      (温度约束)

其中:
    C_max = makespan（总完成时间）
    E_total = sum_j integral_0^{C_max} P_j(t) dt  (总能耗)
    R_total = prod_i P(任务i按时完成)  (系统可靠性)

贪心分区启发式（源自 partition_greedy）：
    将任务按计算量降序排列，依次放入当前负载较轻的处理器。

博弈抢占策略（源自 reversi_game）：
    将调度视为零和博弈：调度器 vs 不确定性。
    在每个决策点，评估所有合法移动（任务迁移）的收益，
    选择翻转最多"劣势"（即减少makespan最多）的移动。
"""

import numpy as np
from utils import rref_matrix


def greedy_partition_load_balance(weights, n_bins):
    """
    贪心算法解决分区问题（源自 partition_greedy）。

    给定权重列表 w，将其分配到 n_bins 个箱子中，
    使得各箱子负载尽可能均衡。

    算法:
        1) 将权重按降序排列
        2) 对每个权重，放入当前总负载最小的箱子

    返回:
        assignment: list of int, 每个权重所属的箱子索引
        bin_loads: ndarray, 各箱子总负载
    """
    weights = np.array(weights, dtype=float)
    n = weights.size
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    sorted_idx = np.argsort(weights)[::-1]
    bin_loads = np.zeros(n_bins, dtype=float)
    assignment = np.zeros(n, dtype=int)
    for idx in sorted_idx:
        min_bin = int(np.argmin(bin_loads))
        assignment[idx] = min_bin
        bin_loads[min_bin] += weights[idx]
    return assignment.tolist(), bin_loads


def solve_task_mapping_ilp(n_tasks, n_procs, cost_matrix, max_solutions=5):
    """
    使用整数线性规划思想（源自 pariomino_tiling_solver 的RREF解法）
    求解小规模任务映射问题。

    由于ILP是NP-hard，此处采用松弛+贪婪修正策略：
        1) 构造约束矩阵 A (覆盖约束 + 容量约束)
        2) 计算RREF以分析解空间结构
        3) 贪婪搜索可行二元解

    参数:
        n_tasks: int
        n_procs: int
        cost_matrix: ndarray, shape (n_tasks, n_procs), 任务i在处理器j上的代价
        max_solutions: int, 返回的最大解数

    返回:
        best_assign: ndarray, shape (n_tasks,), 最优分配
        best_cost: float
    """
    n_vars = n_tasks * n_procs
    # 约束: 每个任务恰好分配到一个处理器 -> n_tasks 个方程
    # 约束: 每个处理器最多 ceil(n_tasks/n_procs) 个任务 -> n_procs 个方程
    max_per_proc = int(np.ceil(n_tasks / n_procs))
    n_eq = n_tasks + n_procs
    A = np.zeros((n_eq, n_vars), dtype=float)
    b = np.zeros(n_eq, dtype=float)

    # 任务覆盖约束
    for i in range(n_tasks):
        for j in range(n_procs):
            A[i, i * n_procs + j] = 1.0
        b[i] = 1.0

    # 处理器容量约束
    for j in range(n_procs):
        for i in range(n_tasks):
            A[n_tasks + j, i * n_procs + j] = 1.0
        b[n_tasks + j] = max_per_proc

    # RREF分析
    Ab = np.column_stack([A, b])
    rref_ab, det = rref_matrix(Ab)
    rank = 0
    for i in range(min(n_eq, n_vars)):
        if np.any(np.abs(rref_ab[i, :n_vars]) > 1e-10):
            rank += 1

    # 贪婪搜索（基于cost_matrix）
    tasks_sorted = np.argsort(np.max(cost_matrix, axis=1))[::-1]
    best_cost = np.inf
    best_assign = None

    def greedy_assign(seed_proc=0):
        assign = np.full(n_tasks, -1, dtype=int)
        proc_load = np.zeros(n_procs, dtype=float)
        proc_count = np.zeros(n_procs, dtype=int)
        rng = np.random.default_rng(seed_proc)
        for i in tasks_sorted:
            # 可选处理器: 未达容量上限且cost不过大的
            feasible = [j for j in range(n_procs)
                        if proc_count[j] < max_per_proc]
            if not feasible:
                return None, np.inf
            # 加权选择: cost越低概率越高
            costs = np.array([cost_matrix[i, j] + 0.1 * proc_load[j] for j in feasible])
            probs = 1.0 / (costs + 1e-9)
            probs = probs / np.sum(probs)
            j = rng.choice(feasible, p=probs)
            assign[i] = j
            proc_load[j] += cost_matrix[i, j]
            proc_count[j] += 1
        total_cost = np.max(proc_load)
        return assign, total_cost

    for seed in range(min(max_solutions, 20)):
        assign, cost = greedy_assign(seed)
        if assign is not None and cost < best_cost:
            best_cost = cost
            best_assign = assign.copy()

    if best_assign is None:
        #  fallback: 简单round-robin
        best_assign = np.arange(n_tasks) % n_procs
        best_cost = float(np.max([np.sum(cost_matrix[best_assign == j, j]) for j in range(n_procs)]))

    return best_assign, float(best_cost)


def reversi_greedy_move(board, player, move_values):
    """
    源自 reversi_game 的贪心最优移动策略。

    将调度状态映射到8x8棋盘：
        board[i,j] = 0 空闲
        board[i,j] = 1 处理器1占用
        board[i,j] = 2 处理器2占用

    move_values[i,j] 表示在位置(i,j)放置任务带来的收益。
    策略:
        - 角落位置永远优先
        - 否则选择收益最大的合法位置

    返回:
        i, j: 最优位置（0-based）
    """
    board = np.array(board, dtype=int)
    move_values = np.array(move_values, dtype=float)
    best_val = -np.inf
    best_ij = (-1, -1)
    m, n = board.shape
    corners = [(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1)]

    for i in range(m):
        for j in range(n):
            if board[i, j] != 0:
                continue
            val = move_values[i, j]
            if (i, j) in corners:
                return i, j
            if val > best_val:
                best_val = val
                best_ij = (i, j)
    return best_ij


def schedule_tasks_greedy(tasks, platform, surrogate=None, alpha_makespan=0.6,
                          alpha_energy=0.3, alpha_reliability=0.1):
    """
    贪心调度主算法。

    步骤:
        1) 用贪心分区将任务初步分配到处理器（负载均衡）
        2) 对每个处理器内部，按截止时间EDD排序
        3) 用博弈策略进行抢占和迁移优化

    参数:
        tasks: list of TaskWorkload
        platform: HeterogeneousPlatform
        surrogate: PerformanceSurrogate or None
        alpha_makespan, alpha_energy, alpha_reliability: 多目标权重

    返回:
        schedule: dict, {proc_id: [(task_id, start, finish), ...]}
        metrics: dict, 调度指标
    """
    n_tasks = len(tasks)
    n_procs = len(platform.processors)

    # 步骤1: 估计每个任务在每个处理器上的执行时间
    cost_matrix = np.zeros((n_tasks, n_procs), dtype=float)
    for i, task in enumerate(tasks):
        for j, proc in enumerate(platform.processors):
            base_time = proc.execution_time(task.base_flops, task.compute_intensity)
            if surrogate is not None:
                feat = np.clip(task.compute_intensity / 10.0, 0.0, 1.0)
                pred_factor = surrogate.predict(np.array([feat]))[0]
                base_time *= max(0.5, min(2.0, pred_factor))
            cost_matrix[i, j] = base_time

    # 步骤2: 贪心分区
    assign, _ = greedy_partition_load_balance(cost_matrix[:, 0], n_procs)
    # 上面的贪心分区只用了第0列，不够准确。改用ILP-based映射
    assign, _ = solve_task_mapping_ilp(n_tasks, n_procs, cost_matrix, max_solutions=10)

    # 步骤3: 每个处理器内部调度（最早截止时间优先 EDD）
    schedule = {j: [] for j in range(n_procs)}
    for j in range(n_procs):
        proc_tasks = [(i, tasks[i]) for i in range(n_tasks) if assign[i] == j]
        # 按截止时间排序
        proc_tasks.sort(key=lambda it: it[1].deadline if it[1].deadline is not None else float('inf'))
        t_now = 0.0
        for i, task in proc_tasks:
            exec_time = cost_matrix[i, j]
            finish = t_now + exec_time
            schedule[j].append((i, t_now, finish))
            t_now = finish

    # 步骤4: 计算指标
    makespan = max(
        max((fin for (_, _, fin) in schedule[j]), default=0.0)
        for j in range(n_procs)
    )
    total_energy = 0.0
    log_reliability = 0.0
    for j in range(n_procs):
        proc = platform.processors[j]
        busy_time = sum(fin - st for (_, st, fin) in schedule[j])
        total_energy += proc.power_idle_w * makespan + (proc.power_peak_w - proc.power_idle_w) * busy_time
        for (i, st, fin) in schedule[j]:
            task = tasks[i]
            allocated = fin - st
            speed_ratio = proc.peak_gflops / max(task.reference_peak_gflops, 1e-12)
            rel = task.reliability_probability(allocated, speed_ratio)
            log_reliability += np.log(max(rel, 1e-15))
    reliability_prod = float(np.exp(log_reliability))

    metrics = {
        'makespan': float(makespan),
        'total_energy_j': float(total_energy),
        'system_reliability': float(reliability_prod),
        'objective': float(alpha_makespan * makespan + alpha_energy * total_energy * 1e-6
                          + alpha_reliability * (1.0 - reliability_prod))
    }

    return schedule, metrics


def local_search_improvement(tasks, platform, schedule, metrics, max_iter=50):
    """
    局部搜索改进：尝试将任务从一个处理器迁移到另一个处理器，
    若目标函数下降则接受。
    """
    n_tasks = len(tasks)
    n_procs = len(platform.processors)
    # 从schedule重建assign
    assign = np.full(n_tasks, -1, dtype=int)
    for j, proc_sched in schedule.items():
        for (i, _, _) in proc_sched:
            assign[i] = j

    best_obj = metrics['objective']
    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(n_tasks):
            curr_j = assign[i]
            for new_j in range(n_procs):
                if new_j == curr_j:
                    continue
                # 尝试迁移
                assign[i] = new_j
                # 快速估算新目标
                proc_loads = np.zeros(n_procs)
                for idx in range(n_tasks):
                    jdx = assign[idx]
                    proc_loads[jdx] += tasks[idx].base_flops / max(platform.processors[jdx].peak_gflops, 1e-6)
                new_makespan = float(np.max(proc_loads))
                if new_makespan < best_obj * 0.99:
                    # 粗略接受（实际应完整重算）
                    best_obj = new_makespan
                    improved = True
                else:
                    assign[i] = curr_j
        if improved:
            # 重建schedule
            schedule = {j: [] for j in range(n_procs)}
            for j in range(n_procs):
                proc_tasks = [(i, tasks[i]) for i in range(n_tasks) if assign[i] == j]
                proc_tasks.sort(key=lambda it: it[1].deadline if it[1].deadline is not None else float('inf'))
                t_now = 0.0
                for i, task in proc_tasks:
                    exec_time = task.base_flops / max(platform.processors[j].peak_gflops, 1e-6)
                    finish = t_now + exec_time
                    schedule[j].append((i, t_now, finish))
                    t_now = finish

    return schedule
