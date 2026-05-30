
import numpy as np
from scipy import integrate


class SteinerbergerVerifier:

    EULER_MASCHERONI = 0.57721566490153286060651209008240243104215933593992

    def __init__(self):
        pass

    @staticmethod
    def steinerberger_function(n: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        value = np.zeros_like(x)
        for k in range(1, n + 1):
            value += np.abs(np.sin(k * np.pi * x)) / k
        return value

    @staticmethod
    def harmonic_number(n: int) -> float:
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")

        if n <= 10000:
            return float(np.sum(1.0 / np.arange(1, n + 1)))


        gamma = SteinerbergerVerifier.EULER_MASCHERONI
        H = (np.log(n) + gamma
             + 1.0 / (2.0 * n)
             - 1.0 / (12.0 * n ** 2)
             + 1.0 / (120.0 * n ** 4)
             - 1.0 / (252.0 * n ** 6))
        return float(H)

    def exact_integral(self, n: int) -> float:
        H_n = self.harmonic_number(n)
        return (2.0 / np.pi) * H_n

    def numerical_integral_trapz(self, n: int, num_points: int = 10000) -> float:
        x = np.linspace(0.0, 1.0, num_points)
        y = self.steinerberger_function(n, x)
        return float(np.trapezoid(y, x))

    def numerical_integral_simpson(self, n: int, num_points: int = 10001) -> float:
        x = np.linspace(0.0, 1.0, num_points)
        y = self.steinerberger_function(n, x)
        return float(integrate.simpson(y, x))

    def numerical_integral_quad(self, n: int) -> float:
        result, err = integrate.quad(
            lambda x: float(self.steinerberger_function(n, np.array([x]))[0]),
            0.0, 1.0, limit=500
        )
        return float(result)

    def verify_integration(self, n_values: list = None) -> dict:
        if n_values is None:
            n_values = [5, 10, 20, 50, 100]

        results = []
        for n in n_values:
            exact = self.exact_integral(n)
            trapz = self.numerical_integral_trapz(n, num_points=20001)
            simpson = self.numerical_integral_simpson(n, num_points=20001)
            quad = self.numerical_integral_quad(n)

            results.append({
                'n': n,
                'exact': exact,
                'trapz': trapz,
                'trapz_error': abs(trapz - exact),
                'simpson': simpson,
                'simpson_error': abs(simpson - exact),
                'quad': quad,
                'quad_error': abs(quad - exact)
            })

        return {'results': results}

    def semantic_embedding_integral_test(self, embedding: np.ndarray,
                                         weight_func, dim: int = 5) -> dict:
        embedding = np.asarray(embedding, dtype=float)

        def integrand(x):
            return float(weight_func(embedding, x))


        trapz_result, _ = integrate.quad(integrand, 0.0, 1.0, limit=100)

        x_gauss, w_gauss = np.polynomial.legendre.leggauss(64)
        x_mapped = 0.5 * (x_gauss + 1.0)
        w_mapped = 0.5 * w_gauss
        gauss_result = np.sum(w_mapped * np.array([integrand(xi) for xi in x_mapped]))

        return {
            'quad_result': trapz_result,
            'gauss_result': gauss_result,
            'difference': abs(trapz_result - gauss_result),
            'relative_error': abs(trapz_result - gauss_result) / (abs(trapz_result) + 1e-15)
        }


def demo():
    print("=" * 60)
    print("Steinerberger数值精度验证演示")
    print("=" * 60)

    verifier = SteinerbergerVerifier()


    print("\n--- 调和数测试 ---")
    for n in [10, 100, 1000]:
        H_direct = float(np.sum(1.0 / np.arange(1, n + 1)))
        H_asymp = verifier.harmonic_number(n)
        print(f"n={n:5d}: 直接求和={H_direct:.10f}, 渐近公式={H_asymp:.10f}, "
              f"偏差={abs(H_direct - H_asymp):.2e}")


    print("\n--- 积分精度验证 ---")
    results = verifier.verify_integration(n_values=[5, 10, 20, 50, 100])
    for r in results['results']:
        print(f"\nn={r['n']}:")
        print(f"  精确值:    {r['exact']:.10f}")
        print(f"  梯形法:    {r['trapz']:.10f}  (误差: {r['trapz_error']:.2e})")
        print(f"  Simpson:   {r['simpson']:.10f}  (误差: {r['simpson_error']:.2e})")
        print(f"  Quad:      {r['quad']:.10f}  (误差: {r['quad_error']:.2e})")


    print("\n--- 语义嵌入积分压力测试 ---")
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(8)

    def semantic_weight_func(emb, x):
        n = min(len(emb), 10)
        basis = np.sin(np.pi * np.arange(1, n + 1) * x)
        return float(np.dot(emb[:n], basis))

    test_result = verifier.semantic_embedding_integral_test(
        embedding, semantic_weight_func
    )
    print(f"Quad结果:  {test_result['quad_result']:.10f}")
    print(f"Gauss结果: {test_result['gauss_result']:.10f}")
    print(f"绝对偏差:  {test_result['difference']:.2e}")
    print(f"相对偏差:  {test_result['relative_error']:.2e}")

    print("\n模块运行完成")
    return verifier, results


if __name__ == "__main__":
    demo()
