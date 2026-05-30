
import numpy as np
from typing import Tuple, Callable
from cosmology import Cosmology


class NBodyIntegrator:

    def __init__(
        self,
        cosmology: Cosmology,
        softening: float = 0.5,
        eta: float = 0.2,
        use_adaptive_step: bool = False,
        tol: float = 1e-4,
    ):
        self.cosmo = cosmology
        self.softening = max(softening, 1e-6)
        self.eta = eta
        self.use_adaptive_step = use_adaptive_step
        self.tol = tol

    def compute_timestep(self, acc: np.ndarray) -> float:
        acc_mag = np.linalg.norm(acc, axis=1)
        max_acc = acc_mag.max()
        if max_acc < 1e-15:
            return 1.0
        dt = self.eta * np.sqrt(self.softening / max_acc)

        dt = np.clip(dt, 1e-5, 1.0)
        return dt

    def drift_step(
        self, pos: np.ndarray, vel: np.ndarray, dt: float, L: float
    ) -> np.ndarray:
        pos_new = pos + vel * dt
        pos_new = pos_new % L
        return pos_new

    def kick_step(self, vel: np.ndarray, acc: np.ndarray, dt: float) -> np.ndarray:
        return vel + acc * dt

    def leapfrog_step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        dt: float,
        L: float,
        compute_acc: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        raise NotImplementedError("请实现 leapfrog_step 方法")

    def rk12_step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        dt: float,
        L: float,
        compute_acc: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        n_part = pos.shape[0]


        acc0 = compute_acc(pos)
        pos_euler = (pos + vel * dt) % L
        vel_euler = vel + acc0 * dt


        pos_pred = (pos + vel * dt) % L
        acc_pred = compute_acc(pos_pred)
        vel_pred = vel + acc_pred * dt
        pos_heun = (pos + 0.5 * (vel + vel_pred) * dt) % L
        vel_heun = vel + 0.5 * (acc0 + acc_pred) * dt


        err_pos = np.linalg.norm(pos_heun - pos_euler, axis=1).max()
        err_vel = np.linalg.norm(vel_heun - vel_euler, axis=1).max()
        error = max(err_pos, err_vel)

        return pos_heun, vel_heun, acc_pred, error

    def evolve(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        t_span: Tuple[float, float],
        L: float,
        compute_acc: Callable[[np.ndarray], np.ndarray],
        n_steps: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t0, t1 = t_span
        pos = pos0.copy()
        vel = vel0.copy()
        acc = compute_acc(pos)

        if self.use_adaptive_step:

            t = t0
            t_list = [t]
            pos_list = [pos.copy()]
            vel_list = [vel.copy()]
            acc_list = [acc.copy()]
            max_steps = 10000
            step_count = 0
            while t < t1 and step_count < max_steps:
                dt = self.compute_timestep(acc)
                dt = min(dt, t1 - t)
                pos_new, vel_new, acc_new, err = self.rk12_step(
                    pos, vel, dt, L, compute_acc
                )

                if err > self.tol and dt > 1e-5:
                    dt = dt * 0.5
                    continue
                pos, vel, acc = pos_new, vel_new, acc_new
                t += dt

                if err < self.tol * 0.1:
                    dt = min(dt * 2.0, 1.0)
                t_list.append(t)
                pos_list.append(pos.copy())
                vel_list.append(vel.copy())
                acc_list.append(acc.copy())
                step_count += 1
            return (
                np.array(t_list),
                np.array(pos_list),
                np.array(vel_list),
                np.array(acc_list),
            )
        else:

            if n_steps is None:
                n_steps = 100
            dt = (t1 - t0) / n_steps
            t_arr = np.zeros(n_steps + 1)
            pos_arr = np.zeros((n_steps + 1, pos0.shape[0], 3))
            vel_arr = np.zeros((n_steps + 1, pos0.shape[0], 3))
            acc_arr = np.zeros((n_steps + 1, pos0.shape[0], 3))
            t_arr[0] = t0
            pos_arr[0] = pos
            vel_arr[0] = vel
            acc_arr[0] = acc
            for i in range(n_steps):
                pos, vel, acc = self.leapfrog_step(pos, vel, acc, dt, L, compute_acc)
                t_arr[i + 1] = t0 + (i + 1) * dt
                pos_arr[i + 1] = pos
                vel_arr[i + 1] = vel
                acc_arr[i + 1] = acc
            return t_arr, pos_arr, vel_arr, acc_arr


def total_energy(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    phi: np.ndarray,
    pm_solver,
) -> Tuple[float, float, float]:
    K = 0.5 * np.sum(mass[:, None] * vel ** 2)

    N = pm_solver.N
    L = pm_solver.L
    idx = ((pos / L) * N).astype(int) % N
    phi_p = phi[idx[:, 0], idx[:, 1], idx[:, 2]]
    U = 0.5 * np.sum(mass * phi_p)
    return K + U, K, U


if __name__ == "__main__":
    from pm_solver import PMSolver
    from cosmology import Cosmology

    cosmo = Cosmology()
    N = 16
    L = 100.0
    solver = PMSolver(N, L)
    n_part = N ** 3
    pos = np.random.rand(n_part, 3) * L
    vel = np.random.randn(n_part, 3) * 1e-3
    mass = np.ones(n_part) * 1e10
    rho_mean = n_part * 1e10 / (L ** 3)

    def get_acc(p):
        return solver.compute_gravity(p, mass, rho_mean)

    integrator = NBodyIntegrator(cosmo, use_adaptive_step=False)
    t_arr, pos_arr, vel_arr, acc_arr = integrator.evolve(
        pos, vel, (0.0, 1.0), L, get_acc, n_steps=50
    )
    print(f"演化完成: {len(t_arr)} 步")
    print(f"最终位置均值: {pos_arr[-1].mean(axis=0)}")
