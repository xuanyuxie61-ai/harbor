"""
数值精度验证模块：语义嵌入计算的数值压力测试

原项目映射: 1161_steinerberger
科学背景: Steinerberger函数:
            f(n, x) = sum_{k=1}^n |sin(pi*k*x)| / k
          
          该函数具有大量局部极值，是数值积分软件的压力测试函数。
          
          积分:
            I(n) = integral_0^1 f(n, x) dx
                 = sum_{k=1}^n (1/k) * integral_0^1 |sin(pi*k*x)| dx
                 = sum_{k=1}^n (1/k) * (2/pi)
                 = (2/pi) * H_n
          其中 H_n 是第 n 个调和数。

数学模型:
    调和数:
        H_n = sum_{k=1}^n 1/k
        
    渐近展开:
        H_n = ln(n) + gamma + 1/(2n) - 1/(12*n^2) + 1/(120*n^4) - ...
        其中 gamma = 0.5772156649... (Euler-Mascheroni常数)

在NLP语义嵌入中的应用:
    验证语义嵌入相关数值积分（如注意力权重积分、
    嵌入空间测度计算等）的精度和稳定性。
"""

import numpy as np
from scipy import integrate


class SteinerbergerVerifier:
    """
    Steinerberger数值精度验证器。
    """

    EULER_MASCHERONI = 0.57721566490153286060651209008240243104215933593992

    def __init__(self):
        pass

    @staticmethod
    def steinerberger_function(n: int, x: np.ndarray) -> np.ndarray:
        """
        计算Steinerberger函数值。
        
        f(n, x) = sum_{k=1}^n |sin(pi*k*x)| / k
        """
        x = np.asarray(x, dtype=float)
        value = np.zeros_like(x)
        for k in range(1, n + 1):
            value += np.abs(np.sin(k * np.pi * x)) / k
        return value

    @staticmethod
    def harmonic_number(n: int) -> float:
        """
        计算调和数 H_n。
        
        对小的 n 使用直接求和，对大的 n 使用渐近展开:
            H_n = ln(n) + gamma + 1/(2n) - 1/(12*n^2) + 1/(120*n^4) - 1/(252*n^6) + ...
        """
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")

        if n <= 10000:
            return float(np.sum(1.0 / np.arange(1, n + 1)))

        # 渐近展开
        gamma = SteinerbergerVerifier.EULER_MASCHERONI
        H = (np.log(n) + gamma
             + 1.0 / (2.0 * n)
             - 1.0 / (12.0 * n ** 2)
             + 1.0 / (120.0 * n ** 4)
             - 1.0 / (252.0 * n ** 6))
        return float(H)

    def exact_integral(self, n: int) -> float:
        """
        计算精确积分值。
        
        I(n) = (2/pi) * H_n
        """
        H_n = self.harmonic_number(n)
        return (2.0 / np.pi) * H_n

    def numerical_integral_trapz(self, n: int, num_points: int = 10000) -> float:
        """
        使用梯形法则数值积分。
        """
        x = np.linspace(0.0, 1.0, num_points)
        y = self.steinerberger_function(n, x)
        return float(np.trapezoid(y, x))

    def numerical_integral_simpson(self, n: int, num_points: int = 10001) -> float:
        """
        使用Simpson法则数值积分。
        """
        x = np.linspace(0.0, 1.0, num_points)
        y = self.steinerberger_function(n, x)
        return float(integrate.simpson(y, x))

    def numerical_integral_quad(self, n: int) -> float:
        """
        使用scipy.quad自适应积分。
        """
        result, err = integrate.quad(
            lambda x: float(self.steinerberger_function(n, np.array([x]))[0]),
            0.0, 1.0, limit=500
        )
        return float(result)

    def verify_integration(self, n_values: list = None) -> dict:
        """
        验证不同n值下的数值积分精度。
        """
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
        """
        对语义嵌入的数值积分进行压力测试。
        
        计算: integral_{0}^{1} weight_func(embedding, x) dx
        使用多种数值方法并比较结果。
        """
        embedding = np.asarray(embedding, dtype=float)

        def integrand(x):
            return float(weight_func(embedding, x))

        # 多种数值方法
        trapz_result, _ = integrate.quad(integrand, 0.0, 1.0, limit=100)
        # 使用固定点高斯积分
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
    """模块功能演示"""
    print("=" * 60)
    print("Steinerberger数值精度验证演示")
    print("=" * 60)

    verifier = SteinerbergerVerifier()

    # 调和数测试
    print("\n--- 调和数测试 ---")
    for n in [10, 100, 1000]:
        H_direct = float(np.sum(1.0 / np.arange(1, n + 1)))
        H_asymp = verifier.harmonic_number(n)
        print(f"n={n:5d}: 直接求和={H_direct:.10f}, 渐近公式={H_asymp:.10f}, "
              f"偏差={abs(H_direct - H_asymp):.2e}")

    # 积分精度验证
    print("\n--- 积分精度验证 ---")
    results = verifier.verify_integration(n_values=[5, 10, 20, 50, 100])
    for r in results['results']:
        print(f"\nn={r['n']}:")
        print(f"  精确值:    {r['exact']:.10f}")
        print(f"  梯形法:    {r['trapz']:.10f}  (误差: {r['trapz_error']:.2e})")
        print(f"  Simpson:   {r['simpson']:.10f}  (误差: {r['simpson_error']:.2e})")
        print(f"  Quad:      {r['quad']:.10f}  (误差: {r['quad_error']:.2e})")

    # 语义嵌入积分测试
    print("\n--- 语义嵌入积分压力测试 ---")
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(8)

    def semantic_weight_func(emb, x):
        """语义权重函数：嵌入与正弦基的内积"""
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
