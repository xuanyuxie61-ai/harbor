
import numpy as np
import time
from checkpoint_tree import CheckpointTree, build_default_tree
from fault_model import FaultPredictor, GammaFaultModel
from state_compression import svd_compress, svd_reconstruct, optimal_rank, compress_state_trig, reconstruct_state_trig
from recovery_mdp import CheckpointMDP


class CheckpointManager:

    def __init__(self, tree: CheckpointTree = None,
                 predictor: FaultPredictor = None,
                 compression_method: str = "svd",
                 target_compression_ratio: float = 0.1):
        if tree is None:
            tree = build_default_tree()
        self.tree = tree
        if predictor is None:
            predictor = FaultPredictor(significance=0.05)
        self.predictor = predictor
        self.compression_method = compression_method
        self.target_compression_ratio = target_compression_ratio
        self.checkpoints = {}
        self.fault_history = []
        self.total_wasted_time = 0.0
        self.total_checkpoint_time = 0.0

    def create_checkpoint(self, step: int, state: np.ndarray, level: int = 1) -> dict:
        meta = {"step": step, "level": level, "original_size": state.size}
        if self.compression_method == "svd":
            rank = optimal_rank(state, energy_threshold=0.95)

            if state.ndim == 1:
                max_rank = max(1, int(len(state) * self.target_compression_ratio))
            else:
                n, m = state.shape
                max_rank = max(1, int(state.size * self.target_compression_ratio / (n + m + 1)))
            rank = min(rank, max_rank)
            if state.ndim == 1:
                state_mat = state.reshape(-1, 1)
                U, s, Vt, _ = svd_compress(state_mat, rank)
                meta["type"] = "svd"
                meta["U"] = U
                meta["s"] = s
                meta["Vt"] = Vt
                meta["shape"] = state_mat.shape
            else:
                U, s, Vt, _ = svd_compress(state, rank)
                meta["type"] = "svd"
                meta["U"] = U
                meta["s"] = s
                meta["Vt"] = Vt
                meta["shape"] = state.shape
        elif self.compression_method == "trig":
            n_coarse = max(4, int(len(state) * self.target_compression_ratio))
            xd, yd, N = compress_state_trig(state, n_coarse=n_coarse)
            meta["type"] = "trig"
            meta["xd"] = xd
            meta["yd"] = yd
            meta["N"] = N
        else:
            meta["type"] = "raw"
            meta["state"] = state.copy()

        if step not in self.checkpoints:
            self.checkpoints[step] = {}
        self.checkpoints[step][level] = meta
        return meta

    def restore_checkpoint(self, step: int, level: int = None) -> np.ndarray:
        if step not in self.checkpoints:
            raise ValueError(f"No checkpoint at step {step}")
        available = self.checkpoints[step]
        if level is None:
            level = min(available.keys())
        meta = available[level]
        if meta["type"] == "svd":
            state = svd_reconstruct(meta["U"], meta["s"], meta["Vt"])
            if state.shape[1] == 1:
                state = state.ravel()
            return state
        elif meta["type"] == "trig":
            return reconstruct_state_trig(meta["xd"], meta["yd"], meta["N"])
        else:
            return meta["state"].copy()

    def find_latest_checkpoint(self, current_step: int) -> int:
        steps = [s for s in self.checkpoints.keys() if s <= current_step]
        if not steps:
            return None
        return max(steps)

    def simulate_fault_and_recover(self, current_step: int, current_state: np.ndarray,
                                   fault_model: GammaFaultModel) -> tuple:
        ck_step = self.find_latest_checkpoint(current_step)
        if ck_step is None:

            return np.zeros_like(current_state), 0, -1

        available = self.checkpoints[ck_step]
        if 0 in available:
            level = 0
        elif 1 in available:
            level = 1
        else:
            level = min(available.keys())
        recovered = self.restore_checkpoint(ck_step, level)
        wasted = current_step - ck_step
        self.total_wasted_time += wasted
        return recovered, ck_step, level

    def adaptive_interval(self, base_interval: float = 50.0) -> float:
        rec = self.predictor.recommended_checkpoint_interval(safety_factor=0.8)

        return max(5.0, min(base_interval, rec))

    def compression_error(self, step: int, true_state: np.ndarray) -> float:
        if step not in self.checkpoints:
            return 0.0
        levels = self.checkpoints[step]
        if not levels:
            return 0.0
        meta = levels[min(levels.keys())]
        if meta["type"] == "raw":
            return 0.0
        restored = self.restore_checkpoint(step)
        diff = restored - true_state
        norm_true = np.linalg.norm(true_state)
        if norm_true < 1.0e-14:
            return np.linalg.norm(diff)
        return np.linalg.norm(diff) / norm_true
