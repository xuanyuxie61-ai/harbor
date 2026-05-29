"""
parallel_circuit_simulator.py
基于项目 237_cuda_loop (CUDA并行调度) 与 845_pagerank2 (稀疏矩阵/图论)
的量子电路并行模拟器。

核心数学模型:
1. CUDA 并行调度模拟:
   Grid 维度: (blocks_x, blocks_y, blocks_z)
   Block 维度: (threads_x, threads_y, threads_z)
   线性索引: K = tx + blockDim.x*ty + blockDim.x*blockDim.y*tz
              + blockDim.x*blockDim.y*blockDim.z*bx
              + blockDim.x*blockDim.y*blockDim.z*gridDim.x*by
              + blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*bz
   总线程数: chunk = product(gridDim) * product(blockDim)
   循环分配: T = K, while T < N: execute_task(T), T += chunk

2. 量子门操作的张量积表示:
   单量子门: U_i = I^{⊗(i-1)} ⊗ U ⊗ I^{⊗(n-i)}
   双量子门: U_{ij} = I^{⊗(i-1)} ⊗ |0><0| ⊗ I^{⊗(j-i-1)} ⊗ I ⊗ ...
              + I^{⊗(i-1)} ⊗ |1><1| ⊗ I^{⊗(j-i-1)} ⊗ U ⊗ ...

3. PageRank 转移矩阵 (用于量子随机行走的谱分析):
   P_{ji} = A_{ij} / d_i, 其中 d_i 为节点 i 的出度
   Google 矩阵: G = alpha*P + (1-alpha)/N * 1*1^T
   特征值问题: G r = r
"""

import numpy as np
from typing import Tuple, List, Callable, Optional


class ParallelTaskScheduler:
    """
    模拟 CUDA 并行调度策略的任务调度器。
    将大量量子门操作均匀分配给有限数量的"线程"。
    """

    def __init__(
        self,
        blocks: Tuple[int, int, int] = (2, 1, 1),
        threads: Tuple[int, int, int] = (4, 2, 1)
    ):
        self.blocks = blocks
        self.threads = threads
        self.chunk = blocks[0] * blocks[1] * blocks[2] * threads[0] * threads[1] * threads[2]

    def get_task_assignments(self, n_tasks: int) -> List[List[int]]:
        """
        将 n_tasks 个任务按循环分配策略分配给各线程。
        返回: 每个线程被分配的任务索引列表。
        """
        if n_tasks < 0:
            raise ValueError("n_tasks must be non-negative")

        assignments = [[] for _ in range(self.chunk)]
        for task_id in range(n_tasks):
            thread_id = task_id % self.chunk
            assignments[thread_id].append(task_id)
        return assignments

    def schedule_execution(
        self,
        tasks: List[Callable],
        executor: Optional[Callable] = None
    ) -> List:
        """
        调度并执行任务列表。
        由于 Python 没有真正的 GPU 并行，这里模拟调度逻辑。
        """
        if not tasks:
            return []

        assignments = self.get_task_assignments(len(tasks))
        results = [None] * len(tasks)

        for thread_id, task_ids in enumerate(assignments):
            for tid in task_ids:
                if executor is not None:
                    results[tid] = executor(tasks[tid])
                else:
                    results[tid] = tasks[tid]()

        return results


def single_qubit_gate_tensor(
    U: np.ndarray,
    target: int,
    n_qubits: int
) -> np.ndarray:
    """
    构造单量子门 U 在第 target 个量子比特上的张量积表示。
    U_total = I^{⊗target} ⊗ U ⊗ I^{⊗(n_qubits-target-1)}
    """
    if U.shape != (2, 2):
        raise ValueError("U must be a 2x2 matrix")
    if not (0 <= target < n_qubits):
        raise ValueError("target qubit index out of range")

    I2 = np.eye(2, dtype=np.complex128)
    result = np.array([[1.0]], dtype=np.complex128)

    for q in range(n_qubits):
        if q == target:
            result = np.kron(result, U)
        else:
            result = np.kron(result, I2)

    return result


def two_qubit_gate_tensor(
    U: np.ndarray,
    control: int,
    target: int,
    n_qubits: int
) -> np.ndarray:
    """
    构造受控门 U_{control,target} 的张量积表示。
    控制门分解:
    CU = |0><0|_c ⊗ I_t + |1><1|_c ⊗ U_t
    """
    if U.shape != (2, 2):
        raise ValueError("U must be a 2x2 matrix")
    if not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError("Qubit indices out of range")
    if control == target:
        raise ValueError("Control and target must be different")

    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    # 先处理 control 和 target 之间的量子比特
    qubits = list(range(n_qubits))
    result = np.eye(2 ** n_qubits, dtype=np.complex128)

    # 简化实现: 逐位构建
    # |0><0|_c ⊗ I^{⊗(target-control-1)} ⊗ I_t
    # + |1><1|_c ⊗ I^{⊗(target-control-1)} ⊗ U_t

    def build_term(proj_c, gate_t):
        term = np.array([[1.0]], dtype=np.complex128)
        for q in range(n_qubits):
            if q == control:
                term = np.kron(term, proj_c)
            elif q == target:
                term = np.kron(term, gate_t)
            else:
                term = np.kron(term, I2)
        return term

    result = build_term(P0, I2) + build_term(P1, U)
    return result


def apply_quantum_circuit(
    state: np.ndarray,
    gates: List[np.ndarray],
    scheduler: Optional[ParallelTaskScheduler] = None
) -> np.ndarray:
    """
    将一系列量子门依次作用于量子态。
    若提供 scheduler，则模拟并行调度 (顺序执行但在逻辑上分组)。
    """
    current_state = state.copy()
    dim = len(state)

    for gate in gates:
        if gate.shape != (dim, dim):
            raise ValueError(f"Gate dimension {gate.shape} does not match state dimension {dim}")
        current_state = gate @ current_state

    return current_state


def sparse_pagerank_matrix(
    adjacency_list: List[List[int]],
    n_nodes: int,
    damping: float = 0.85
) -> np.ndarray:
    """
    从邻接表构造 PageRank 转移矩阵 (稠密版本，用于小规模量子系统)。

    参数:
        adjacency_list: 每个节点的出边目标节点列表
        n_nodes: 节点总数
        damping: 阻尼系数 alpha

    返回:
        Google 矩阵 G = alpha*P + (1-alpha)/N * J
    """
    if len(adjacency_list) != n_nodes:
        raise ValueError("Adjacency list length must match n_nodes")
    if not (0.0 < damping <= 1.0):
        raise ValueError("Damping must be in (0, 1]")

    P = np.zeros((n_nodes, n_nodes))

    for i, neighbors in enumerate(adjacency_list):
        d_i = len(neighbors)
        if d_i == 0:
            # 无出边: 均匀分布
            P[:, i] = 1.0 / n_nodes
        else:
            for j in neighbors:
                if not (0 <= j < n_nodes):
                    raise ValueError(f"Invalid neighbor index {j}")
                P[j, i] = 1.0 / d_i

    # Google 矩阵
    J = np.ones((n_nodes, n_nodes)) / n_nodes
    G = damping * P + (1.0 - damping) * J

    return G


def power_iteration_pagerank(
    G: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10
) -> np.ndarray:
    """
    使用幂迭代法求解 PageRank 向量。
    r_{k+1} = G @ r_k, 直到收敛。
    """
    n = G.shape[0]
    r = np.ones(n) / n

    for _ in range(max_iter):
        r_new = G @ r
        diff = np.linalg.norm(r_new - r, ord=1)
        r = r_new
        if diff < tol:
            break

    return r


def quantum_circuit_pagerank_spectrum(
    n_qubits: int,
    adjacency_list: List[List[int]],
    damping: float = 0.85
) -> np.ndarray:
    """
    将 PageRank 谱分析应用于量子电路的连通性图。
    返回各量子门的 "重要性排名"。
    """
    n_nodes = len(adjacency_list)
    G = sparse_pagerank_matrix(adjacency_list, n_nodes, damping)
    ranks = power_iteration_pagerank(G)
    return ranks
