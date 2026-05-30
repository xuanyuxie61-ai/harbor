
import numpy as np
from utils import rref_matrix


def greedy_partition_load_balance(weights, n_bins):
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
    n_vars = n_tasks * n_procs


    max_per_proc = int(np.ceil(n_tasks / n_procs))
    n_eq = n_tasks + n_procs
    A = np.zeros((n_eq, n_vars), dtype=float)
    b = np.zeros(n_eq, dtype=float)


    for i in range(n_tasks):
        for j in range(n_procs):
            A[i, i * n_procs + j] = 1.0
        b[i] = 1.0


    for j in range(n_procs):
        for i in range(n_tasks):
            A[n_tasks + j, i * n_procs + j] = 1.0
        b[n_tasks + j] = max_per_proc


    Ab = np.column_stack([A, b])
    rref_ab, det = rref_matrix(Ab)
    rank = 0
    for i in range(min(n_eq, n_vars)):
        if np.any(np.abs(rref_ab[i, :n_vars]) > 1e-10):
            rank += 1


    tasks_sorted = np.argsort(np.max(cost_matrix, axis=1))[::-1]
    best_cost = np.inf
    best_assign = None

    def greedy_assign(seed_proc=0):
        assign = np.full(n_tasks, -1, dtype=int)
        proc_load = np.zeros(n_procs, dtype=float)
        proc_count = np.zeros(n_procs, dtype=int)
        rng = np.random.default_rng(seed_proc)
        for i in tasks_sorted:

            feasible = [j for j in range(n_procs)
                        if proc_count[j] < max_per_proc]
            if not feasible:
                return None, np.inf

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

        best_assign = np.arange(n_tasks) % n_procs
        best_cost = float(np.max([np.sum(cost_matrix[best_assign == j, j]) for j in range(n_procs)]))

    return best_assign, float(best_cost)


def reversi_greedy_move(board, player, move_values):
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
    n_tasks = len(tasks)
    n_procs = len(platform.processors)


    cost_matrix = np.zeros((n_tasks, n_procs), dtype=float)
    for i, task in enumerate(tasks):
        for j, proc in enumerate(platform.processors):
            base_time = proc.execution_time(task.base_flops, task.compute_intensity)
            if surrogate is not None:
                feat = np.clip(task.compute_intensity / 10.0, 0.0, 1.0)
                pred_factor = surrogate.predict(np.array([feat]))[0]
                base_time *= max(0.5, min(2.0, pred_factor))
            cost_matrix[i, j] = base_time


    assign, _ = greedy_partition_load_balance(cost_matrix[:, 0], n_procs)

    assign, _ = solve_task_mapping_ilp(n_tasks, n_procs, cost_matrix, max_solutions=10)


    schedule = {j: [] for j in range(n_procs)}
    for j in range(n_procs):
        proc_tasks = [(i, tasks[i]) for i in range(n_tasks) if assign[i] == j]

        proc_tasks.sort(key=lambda it: it[1].deadline if it[1].deadline is not None else float('inf'))
        t_now = 0.0
        for i, task in proc_tasks:
            exec_time = cost_matrix[i, j]
            finish = t_now + exec_time
            schedule[j].append((i, t_now, finish))
            t_now = finish


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
    n_tasks = len(tasks)
    n_procs = len(platform.processors)

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

                assign[i] = new_j

                proc_loads = np.zeros(n_procs)
                for idx in range(n_tasks):
                    jdx = assign[idx]
                    proc_loads[jdx] += tasks[idx].base_flops / max(platform.processors[jdx].peak_gflops, 1e-6)
                new_makespan = float(np.max(proc_loads))
                if new_makespan < best_obj * 0.99:

                    best_obj = new_makespan
                    improved = True
                else:
                    assign[i] = curr_j
        if improved:

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
