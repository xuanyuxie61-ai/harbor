
import numpy as np


class SemanticSpaceQuantization:

    def __init__(self, n_generators: int = 20, max_iter: int = 100,
                 tol: float = 1e-10):
        if n_generators < 2:
            raise ValueError(f"n_generators must be at least 2, got {n_generators}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}")

        self.n = int(n_generators)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _energy(self, g: np.ndarray) -> float:
        energy = 0.0
        for j in range(1, self.n + 1):
            xl = (g[j - 1] + g[j]) / 2.0
            xr = (g[j + 1] + g[j]) / 2.0
            energy += ((xr - g[j]) ** 3 - (xl - g[j]) ** 3) / 3.0
        return energy

    def _lloyd_step(self, g: np.ndarray) -> np.ndarray:
        g_new = np.zeros(self.n + 2)
        g_new[0] = 0.0
        g_new[self.n + 1] = 1.0


        g_new[1] = (g[0] + 0.5 * (g[1] + g[2])) / 2.0


        for j in range(2, self.n):
            g_new[j] = (0.5 * (g[j - 1] + g[j]) + 0.5 * (g[j] + g[j + 1])) / 2.0


        g_new[self.n] = (0.5 * (g[self.n - 1] + g[self.n]) + g[self.n + 1]) / 2.0

        return g_new

    def quantize(self, init_mode: str = 'random', seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        g = np.zeros(self.n + 2)
        g[0] = 0.0
        g[self.n + 1] = 1.0

        if init_mode == 'random':
            g[1:self.n + 1] = np.sort(rng.random(self.n))
        elif init_mode == 'uniform':
            g[1:self.n + 1] = np.linspace(0.01, 0.99, self.n)
        else:
            raise ValueError(f"init_mode must be 'random' or 'uniform', got {init_mode}")

        energies = []
        motions = []

        for it in range(self.max_iter):
            e = self._energy(g)
            energies.append(e)

            g_new = self._lloyd_step(g)
            motion = np.sum((g_new - g) ** 2) / self.n
            motions.append(motion)

            g = g_new.copy()

            if motion < self.tol:
                break

        return {
            'generators': g[1:self.n + 1].copy(),
            'energy_history': np.array(energies),
            'motion_history': np.array(motions),
            'iterations': it + 1,
            'final_energy': energies[-1]
        }

    def quantize_2d(self, n_generators_x: int, n_generators_y: int,
                    init_mode: str = 'random', seed: int = 42) -> dict:
        cvt_x = SemanticSpaceQuantization(n_generators_x, self.max_iter, self.tol)
        cvt_y = SemanticSpaceQuantization(n_generators_y, self.max_iter, self.tol)

        res_x = cvt_x.quantize(init_mode, seed)
        res_y = cvt_y.quantize(init_mode, seed + 1)


        gx = res_x['generators']
        gy = res_y['generators']
        generators_2d = np.array([[x, y] for x in gx for y in gy])

        return {
            'generators_x': gx,
            'generators_y': gy,
            'generators_2d': generators_2d,
            'energy_x': res_x['final_energy'],
            'energy_y': res_y['final_energy'],
            'iterations_x': res_x['iterations'],
            'iterations_y': res_y['iterations']
        }

    def quantization_error(self, generators: np.ndarray,
                           test_points: np.ndarray) -> float:
        errors = []
        for p in test_points:
            dists = np.abs(p - generators)
            errors.append(np.min(dists) ** 2)
        return np.mean(errors)


def demo():
    print("=" * 60)
    print("语义空间CVT量化演示")
    print("=" * 60)

    cvt = SemanticSpaceQuantization(n_generators=10, max_iter=200, tol=1e-12)
    print(f"\n生成元数: {cvt.n}")
    print(f"最大迭代: {cvt.max_iter}")

    result = cvt.quantize(init_mode='random', seed=42)
    print(f"\n实际迭代次数: {result['iterations']}")
    print(f"最终能量: {result['final_energy']:.6e}")
    print(f"生成元位置:\n{result['generators']}")


    energies = result['energy_history']
    is_decreasing = all(energies[i] >= energies[i + 1] - 1e-14
                        for i in range(len(energies) - 1))
    print(f"\n能量单调递减: {is_decreasing}")


    uniform = np.linspace(0.0, 1.0, cvt.n + 2)[1:-1]
    test_points = np.linspace(0.0, 1.0, 1000)
    error_cvt = cvt.quantization_error(result['generators'], test_points)
    error_uniform = cvt.quantization_error(uniform, test_points)
    print(f"\nCVT量化误差:     {error_cvt:.6e}")
    print(f"均匀量化误差:    {error_uniform:.6e}")
    print(f"改进比例:        {(error_uniform - error_cvt) / error_uniform * 100:.2f}%")


    print("\n--- 二维语义空间CVT ---")
    cvt2 = SemanticSpaceQuantization(n_generators=8, max_iter=200, tol=1e-12)
    res2d = cvt2.quantize_2d(n_generators_x=6, n_generators_y=6, init_mode='random', seed=42)
    print(f"二维生成元数量: {len(res2d['generators_2d'])}")
    print(f"x方向能量: {res2d['energy_x']:.6e}")
    print(f"y方向能量: {res2d['energy_y']:.6e}")

    print("\n模块运行完成")
    return cvt, result


if __name__ == "__main__":
    demo()
