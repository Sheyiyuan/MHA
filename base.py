import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any


class BaseMetropolisHastings(ABC):
    """
    MH算法抽象基类，定义算法核心接口和通用逻辑，子类需实现个性化部分（目标分布、提议分布）。
    """

    def __init__(
        self,
        param_dim: int,
        n_iter: int = 10000,
        burn_in: Optional[int] = None,
        seed: int = 42
    ):
        """
        抽象类初始化：封装通用参数和状态初始化，子类可通过super()调用后补充个性化参数。
        
        Args:
            param_dim: 目标分布的参数维度（1维/高维）
            n_iter: 总迭代次数（马尔可夫链长度）
            burn_in: 燃烧期长度（默认取总迭代次数的10%）
            seed: 随机数种子（确保可复现性）
        """
        # 1. 算法控制参数（通用）
        self.param_dim = param_dim
        self.n_iter = n_iter
        self.burn_in = burn_in if burn_in is not None else n_iter // 10
        self.rng = np.random.default_rng(seed)  # 随机数生成器（可复现）

        # 2. 抽样状态与记录（通用）
        # 先创建 samples 等记录结构，以便子类在 _init_state() 中可以安全访问这些属性
        self.samples: np.ndarray = np.empty((n_iter, param_dim))  # 所有样本（含燃烧期）
        self.accept_flags: list[bool] = []  # 每次迭代的接受标记（True/False）
        self.acceptance_rate: float = 0.0  # 最终接受率（迭代结束后计算）

        # 将当前状态的初始化放到最后，这样子类在覆盖 _init_state 时
        # 可以安全地访问 self.samples 等已创建的属性
        self.current_state: np.ndarray = self._init_state()  # 当前状态（由子类实现或抽象类提供默认）

    # -------------------------------------------------------------------------
    # 抽象方法：子类必须实现，定义个性化逻辑
    # -------------------------------------------------------------------------
    @abstractmethod
    def target_log_prob(self, state: np.ndarray) -> float:
        """
        抽象方法：计算目标分布的对数未归一化概率密度。
        
        Args:
            state: 当前参数状态（形状为 (param_dim,) 的ndarray）
        
        Returns:
            目标分布在该状态下的对数未归一化概率密度（float）
        """
        pass

    @abstractmethod
    def proposal_generate(self, current_state: np.ndarray) -> np.ndarray:
        """
        抽象方法：根据提议分布生成候选状态。
        子类需实现具体的提议分布（如高斯提议、均匀提议）。
        
        Args:
            current_state: 当前参数状态（形状为 (param_dim,) 的ndarray）
        
        Returns:
            生成的候选状态（形状为 (param_dim,) 的ndarray）
        """
        pass

    @abstractmethod
    def proposal_log_prob_ratio(self, current_state: np.ndarray, candidate_state: np.ndarray) -> float:
        """
        抽象方法：计算提议分布的对数概率比 log[Q(current|candidate) / Q(candidate|current)]。
        - 对称提议（如高斯）：该值为0，子类可直接返回0.0；
        - 非对称提议（如随机游走以外的分布）：需按提议分布公式计算。
        
        Args:
            current_state: 当前参数状态
            candidate_state: 候选参数状态
        
        Returns:
            提议分布的对数概率比（float）
        """
        pass

    # -------------------------------------------------------------------------
    # 可重写方法：子类可根据需求覆盖，抽象类提供默认实现
    # -------------------------------------------------------------------------
    def _init_state(self) -> np.ndarray:
        """
        初始化马尔可夫链的初始状态（默认从标准正态分布生成）。
        子类可覆盖该方法（如从先验分布采样初始化）。
        """
        return self.rng.normal(loc=0.0, scale=1.0, size=self.param_dim)

    # -------------------------------------------------------------------------
    # 具体方法：抽象类封装通用逻辑，子类无需实现
    # -------------------------------------------------------------------------
    def _compute_accept_prob(self, current_log_prob: float, candidate_log_prob: float, candidate_state: np.ndarray) -> float:
        """
        通用逻辑：计算候选状态的接受概率（MH核心公式）。
        """
        # 接受概率的对数形式：log(min(1, [pi(candidate)/pi(current)] * [Q(current|candidate)/Q(candidate|current)]))
        log_accept_ratio = candidate_log_prob - current_log_prob + self.proposal_log_prob_ratio(self.current_state, candidate_state)
        # 指数化并取min(1)（避免数值问题）
        return min(1.0, np.exp(log_accept_ratio))

    def _step(self) -> None:
        """
        通用逻辑：单步迭代（生成候选→计算接受概率→决定状态→记录结果）。
        是run()方法的核心步骤，子类无需修改。
        """
        # 1. 生成候选状态
        candidate_state = self.proposal_generate(self.current_state)

        # 2. 计算当前状态和候选状态的目标分布对数概率
        current_log_prob = self.target_log_prob(self.current_state)
        candidate_log_prob = self.target_log_prob(candidate_state)

        # 3. 计算接受概率
        accept_prob = self._compute_accept_prob(current_log_prob, candidate_log_prob, candidate_state)

        # 4. 决定是否接受候选状态
        u = self.rng.uniform(low=0.0, high=1.0)
        accept = u <= accept_prob
        if accept:
            self.current_state = candidate_state.copy()

        # 5. 记录当前状态和接受标记
        self.samples[self._current_iter] = self.current_state
        self.accept_flags.append(accept)

    def run(self) -> None:
        """
        通用逻辑：完整迭代流程（初始化→循环单步迭代→计算接受率）。
        子类直接调用该方法启动抽样，无需修改。
        """
        # 初始化迭代计数
        self._current_iter = 0

        # 循环执行单步迭代
        while self._current_iter < self.n_iter:
            self._step()
            self._current_iter += 1

        # 迭代结束后计算接受率
        self.acceptance_rate = float(np.mean(self.accept_flags))

    def get_effective_samples(self) -> np.ndarray:
        """
        通用逻辑：获取燃烧期后的有效样本（算法核心输出）。
        """
        return self.samples[self.burn_in:]

    def get_summary(self) -> dict[str, Any]:
        """
        通用逻辑：生成抽样结果摘要（便于用户快速查看）。
        """
        effective_samples = self.get_effective_samples()
        return {
            "param_dim": self.param_dim,
            "n_iter": self.n_iter,
            "burn_in": self.burn_in,
            "effective_samples_count": len(effective_samples),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "sample_mean": np.round(np.mean(effective_samples, axis=0), 4),  # 样本均值
            "sample_std": np.round(np.std(effective_samples, axis=0), 4)    # 样本标准差
        }