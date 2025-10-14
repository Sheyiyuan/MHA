import numpy as np

from base import BaseMetropolisHastings


class GaussianMH(BaseMetropolisHastings):
    """
    基于高斯提议分布的MH算法，用于1维双峰分布抽样。
    """
    def __init__(self, proposal_sigma: float = 1.0, **kwargs):
        # 调用父类初始化（传入param_dim、n_iter等通用参数）
        super().__init__(**kwargs)
        # 补充子类特有的参数（高斯提议的标准差）
        self.proposal_sigma = proposal_sigma

    # 实现抽象方法1：目标分布（1维双峰分布的对数未归一化概率）
    def target_log_prob(self, state: np.ndarray) -> float:
        x = state[0]  # 1维参数，取第一个元素
        return -((x - 2) ** 2) / 2 - ((x + 2) ** 2) / 8  # 双峰分布的对数未归一化密度

    # 实现抽象方法2：高斯提议生成候选状态（当前状态+高斯噪声）
    def proposal_generate(self, current_state: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(loc=0.0, scale=self.proposal_sigma, size=self.param_dim)
        return current_state + noise

    # 实现抽象方法3：高斯提议是对称的，对数概率比为0
    def proposal_log_prob_ratio(self, current_state: np.ndarray, candidate_state: np.ndarray) -> float:
        return 0.0


# 使用子类进行抽样
if __name__ == "__main__":
    # 初始化子类（1维参数，10000次迭代，1000次燃烧期）
    mh = GaussianMH(
        param_dim=1,
        n_iter=10000,
        burn_in=1000,
        seed=114514,
        proposal_sigma=1.5  # 高斯提议的标准差（子类特有参数）
    )

    # 启动抽样
    mh.run()

    # 查看结果摘要
    print("抽样结果摘要：")
    for key, value in mh.get_summary().items():
        print(f"{key}: {value}")

    # 查看有效样本（燃烧期后）
    effective_samples = mh.get_effective_samples()
    print(f"\n有效样本前100个：{effective_samples[:100].flatten()}")