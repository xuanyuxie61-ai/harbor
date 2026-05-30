
import numpy as np
from typing import Tuple, List, Callable, Optional


class ParallelTaskScheduler:

    def __init__(
        self,
        blocks: Tuple[int, int, int] = (2, 1, 1),
        threads: Tuple[int, int, int] = (4, 2, 1)
    ):
        self.blocks = blocks
        self.threads = threads
        self.chunk = blocks[0] * blocks[1] * blocks[2] * threads[0] * threads[1] * threads[2]

    def get_task_assignments(self, n_tasks: int) -> List[List[int]]:
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
    if U.shape != (2, 2):
        raise ValueError("U must be a 2x2 matrix")
    if not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError("Qubit indices out of range")
    if control == target:
        raise ValueError("Control and target must be different")

    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)


    qubits = list(range(n_qubits))
    result = np.eye(2 ** n_qubits, dtype=np.complex128)





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
    if len(adjacency_list) != n_nodes:
        raise ValueError("Adjacency list length must match n_nodes")
    if not (0.0 < damping <= 1.0):
        raise ValueError("Damping must be in (0, 1]")

    P = np.zeros((n_nodes, n_nodes))

    for i, neighbors in enumerate(adjacency_list):
        d_i = len(neighbors)
        if d_i == 0:

            P[:, i] = 1.0 / n_nodes
        else:
            for j in neighbors:
                if not (0 <= j < n_nodes):
                    raise ValueError(f"Invalid neighbor index {j}")
                P[j, i] = 1.0 / d_i


    J = np.ones((n_nodes, n_nodes)) / n_nodes
    G = damping * P + (1.0 - damping) * J

    return G


def power_iteration_pagerank(
    G: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10
) -> np.ndarray:
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
    n_nodes = len(adjacency_list)
    G = sparse_pagerank_matrix(adjacency_list, n_nodes, damping)
    ranks = power_iteration_pagerank(G)
    return ranks
