"""
monte_carlo_platelet.py
基于 534_high_card_simulation 的最优停止策略与蒙特卡洛模拟思想，
构建血小板在血管损伤处的随机激活与最优粘附模型。

科学背景：
    血小板在血管损伤处的粘附是一个随机过程。
    每个血小板依次经过损伤区域，根据局部凝血酶浓度决定是否激活粘附。
    类似于"最优停止问题"：血小板需要在观察了一定数量的候选位置后，
    在最佳时机激活以最大化 clot 的稳定性。

数学模型：
    1. 血小板序列：N 个血小板依次到达，每个血小板有激活潜力值 X_i ~ U(0,1)
    2. 最优停止策略：观察前 k 个后，选择第一个超过 max(X_1,...,X_k) 的
    3. 最优 k* ≈ N/e （1/e 法则）

    扩展到血凝场景：
        - 潜力值受局部 IIa 浓度调制：X_i = f(IIa_local)
        - clot 稳定性评分：
            S = Σ_{粘附血小板} w_i * X_i - λ * (N_粘附 - N_最优)²
"""

import numpy as np


class PlateletMonteCarlo:
    """
    血小板粘附的蒙特卡洛模拟器。
    """

    def __init__(self, n_platelets=1000, local_iia=20.0, lambda_penalty=0.01, seed=42):
        """
        参数:
            n_platelets   : int, 血小板总数
            local_iia     : float, 局部凝血酶浓度 (nM)
            lambda_penalty: float, 过度粘附惩罚系数
            seed          : int, 随机种子
        """
        if n_platelets < 2:
            raise ValueError("n_platelets 必须 >= 2")
        self.n = n_platelets
        self.local_iia = max(local_iia, 0.0)
        self.lambda_pen = lambda_penalty
        self.rng = np.random.default_rng(seed)

    def _generate_potentials(self):
        """
        生成血小板激活潜力值。
        潜力值与局部 IIa 浓度正相关：
            X_i ~ Beta(α, β), 其中 α ∝ IIa
        """
        alpha = 1.0 + 0.1 * self.local_iia
        beta = 2.0
        alpha = max(alpha, 0.1)
        beta = max(beta, 0.1)
        x = self.rng.beta(alpha, beta, size=self.n)
        return x

    def optimal_stopping_strategy(self, potentials, skip_num=None):
        """
        基于 534_high_card_simulation 的最优停止策略：
            观察前 skip_num 个，然后选择第一个超过之前最大值的。

        参数:
            potentials : ndarray, 潜力值序列
            skip_num   : int, 跳过数量。None 时使用 N/e 最优值。

        返回:
            selected_idx : int, 选择的索引（-1 表示未选择）
            selected_val : float, 选择的值
        """
        if skip_num is None:
            skip_num = int(round(self.n / np.e))
        skip_num = max(0, min(skip_num, self.n - 1))

        if skip_num == 0:
            return 0, potentials[0]

        threshold = np.max(potentials[:skip_num])
        selected_idx = -1
        selected_val = potentials[-1]  # 默认选最后一个

        for i in range(skip_num, self.n):
            if potentials[i] > threshold:
                selected_idx = i
                selected_val = potentials[i]
                break

        return selected_idx, selected_val

    def simulate_clot_formation(self, n_trials=500, skip_num=None):
        """
        蒙特卡洛模拟 clot 形成过程。

        对每次试验：
            1. 生成 N 个血小板的潜力值
            2. 使用最优停止策略选择粘附时机
            3. 计算 clot 稳定性评分

        返回:
            mean_score : float, 平均稳定性评分
            scores     : ndarray, 每次试验的评分
        """
        scores = np.zeros(n_trials)
        for t in range(n_trials):
            pots = self._generate_potentials()
            idx, val = self.optimal_stopping_strategy(pots, skip_num)

            # 计算 clot 评分：粘附血小板数量 + 平均潜力
            # 简化为：选择前 idx 个中的最优值累积
            if idx >= 0:
                n_adhered = idx + 1
                avg_pot = np.mean(pots[:idx + 1])
            else:
                n_adhered = self.n
                avg_pot = np.mean(pots)

            score = avg_pot * np.sqrt(n_adhered) - self.lambda_pen * (n_adhered - self.n / 3.0) ** 2
            scores[t] = score

        return np.mean(scores), scores

    def find_optimal_skip(self, n_trials=300):
        """
        通过蒙特卡洛搜索最优 skip_num。
        """
        best_skip = 0
        best_score = -np.inf
        results = []
        for skip in range(0, self.n, max(1, self.n // 50)):
            mean_score, _ = self.simulate_clot_formation(n_trials=n_trials, skip_num=skip)
            results.append((skip, mean_score))
            if mean_score > best_score:
                best_score = mean_score
                best_skip = skip
        return best_skip, best_score, results


def demo_optimal_stopping():
    """
    演示最优停止策略在血小板粘附中的应用。
    """
    print("=" * 60)
    print("血小板粘附最优停止策略蒙特卡洛模拟")
    print("=" * 60)

    for iia in [5.0, 20.0, 50.0]:
        mc = PlateletMonteCarlo(n_platelets=500, local_iia=iia, seed=42)
        best_skip, best_score, results = mc.find_optimal_skip(n_trials=200)
        theoretical = int(round(500 / np.e))
        print(f"\n局部 IIa = {iia} nM:")
        print(f"  蒙特卡洛最优 skip = {best_skip}, 平均评分 = {best_score:.4f}")
        print(f"  理论最优 skip (N/e) = {theoretical}")

    return mc


if __name__ == "__main__":
    demo_optimal_stopping()
