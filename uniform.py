import numpy as np
from typing import Callable, Optional, Union

from base import BaseMetropolisHastings

class UniformMH(BaseMetropolisHastings):
    """
    均匀提议分布的MH算法子类实现
    
    特点：
    - 使用均匀分布作为提议分布（对称提议）
    - 适合离散状态空间（如骰子点数、整数变量）
    - 支持自定义步长控制探索范围
    - 兼容连续状态空间
    """
    
    def __init__(
        self,
        target_log_prob: Callable[[np.ndarray], float],
        step_size: Union[float, int],
        param_dim: int,
        state_validator: Optional[Callable[[np.ndarray], bool]] = None,
        discrete: Optional[bool] = None,
        **kwargs
    ):
        """
        初始化均匀提议分布的MH算法
        
        Args:
            target_log_prob: 目标分布的对数未归一化概率函数
            step_size: 步长（控制均匀分布的范围，当前状态±step_size）
            param_dim: 参数维度
            state_validator: 状态验证函数（可选），检查状态是否有效
            **kwargs: 传递给父类的参数（n_iter, burn_in, seed等）
        """
        # 在调用父类构造前，先设置子类需要在 _init_state 中使用的属性
        self.target_log_prob_func = target_log_prob  # 目标分布函数
        self.step_size = step_size                  # 均匀提议的步长
        self.state_validator = state_validator      # 状态验证函数
        # 是否离散：显式指定优先，否则根据 step_size 是否为整数进行启发式判断
        self._is_discrete = bool(discrete) if discrete is not None else isinstance(step_size, int)

        # 调用父类构造（此过程中会调用子类的 _init_state）
        super().__init__(param_dim=param_dim, **kwargs)

        # 若为离散问题：将当前状态四舍五入到整数，并将样本缓冲区改为整数类型
        if self._is_discrete:
            self.current_state = np.round(self.current_state).astype(int)
            self.samples = self.samples.astype(int)
    
    def target_log_prob(self, state: np.ndarray) -> float:
        """实现抽象方法：计算目标分布的对数未归一化概率"""
        return self.target_log_prob_func(state)
    
    def proposal_generate(self, current_state: np.ndarray) -> np.ndarray:
        """
        实现抽象方法：生成均匀提议分布的候选状态
        
        逻辑：在当前状态±step_size范围内均匀采样，对离散状态取整数
        """
        if self._is_discrete:
            # 离散：先生成，再取整，再验证
            candidate_state = current_state
            for _ in range(5):
                noise = self.rng.uniform(
                    low=-self.step_size,
                    high=self.step_size,
                    size=current_state.shape,
                )
                candidate_state = np.round(current_state + noise).astype(int)
                if self.state_validator is None or self.state_validator(candidate_state):
                    return candidate_state
            return candidate_state
        else:
            # 连续：生成连续候选，必要时多次尝试满足验证器
            candidate_state = current_state
            for _ in range(5):
                noise = self.rng.uniform(
                    low=-self.step_size,
                    high=self.step_size,
                    size=current_state.shape,
                )
                candidate_state = current_state + noise
                if self.state_validator is None or self.state_validator(candidate_state):
                    break
            return candidate_state
    
    def proposal_log_prob_ratio(self, current_state: np.ndarray, candidate_state: np.ndarray) -> float:
        """
        实现抽象方法：均匀提议分布的对数概率比
        
        均匀提议是对称分布：Q(x'|x) = Q(x|x')，因此概率比为1，对数比为0
        """
        return 0.0
    
    def _init_state(self) -> np.ndarray:
        """重写初始状态生成方法（适配离散状态约束）"""
        state_validator = getattr(self, "state_validator", None)
        if state_validator is None:
            return super()._init_state()

        # 有验证器时，优先采样满足约束的初值
        for _ in range(100):
            if self._is_discrete:
                # 保守范围：[-5, 5] 整数，若约束更严格，验证器会筛掉
                state = self.rng.integers(low=-5, high=6, size=self.param_dim)
            else:
                state = self.rng.normal(loc=0.0, scale=1.0, size=self.param_dim)
            if state_validator(state):
                return state
        # 多次尝试失败后返回父类默认状态
        return super()._init_state()
    
    def get_summary(self) -> dict:
        """扩展摘要：增加离散状态的频率统计"""
        base_summary = super().get_summary()
        effective_samples = self.get_effective_samples()
        
        # 对1维离散状态计算频率
        if self.param_dim == 1 and self.samples.dtype == int:
            states, counts = np.unique(effective_samples, return_counts=True)
            frequencies = {
                int(state): round(count / len(effective_samples), 4)
                for state, count in zip(states, counts)
            }
            base_summary["state_frequencies"] = frequencies
        else:
            base_summary["state_frequencies"] = {}

        return base_summary


# ------------------------------
# 测试用例：骰子例子验证
# ------------------------------
if __name__ == "__main__":
    # 1. 定义骰子的目标分布（权重：1点=1, 2点=2, 3点=3, 4点=2, 5点=1, 6点=0）
    def dice_target_log_prob(state: np.ndarray) -> float:
        x = int(state[0])
        weights = {1: 1, 2: 2, 3: 3, 4: 2, 5: 1, 6: 0}
        # 加小常数避免log(0)
        return np.log(weights.get(x, 0) + 1e-10)
    
    # 2. 定义骰子状态验证器（必须是1-6的整数）
    def dice_state_validator(state: np.ndarray) -> bool:
        x = state[0]
        return 1 <= x <= 6
    
    # 3. 初始化均匀提议MH算法（步长=1，适合骰子相邻点数跳转）
    mh = UniformMH(
        target_log_prob=dice_target_log_prob,
        step_size=1,
        param_dim=1,
        n_iter=10000,
        burn_in=1000,
        seed=42,
        state_validator=dice_state_validator
    )
    
    # 4. 运行算法
    print("开始运行均匀提议分布的MH算法（骰子例子）...")
    mh.run()
    print("算法运行完成！")
    
    # 5. 输出结果摘要
    summary = mh.get_summary()
    print("\n===== 算法结果摘要 =====")
    print(f"总迭代次数: {summary['n_iter']}")
    print(f"燃烧期长度: {summary['burn_in']}")
    print(f"有效样本数: {summary['effective_samples_count']}")
    print(f"接受率: {summary['acceptance_rate']:.2%}")
    
    # 6. 对比抽样频率与理论频率
    print("\n===== 骰子点数频率对比 =====")
    theoretical = {1: 1/9, 2: 2/9, 3: 3/9, 4: 2/9, 5: 1/9, 6: 0}
    print(f"{'点数':<5} {'抽样频率':<10} {'理论频率':<10}")
    print("-" * 30)
    for point in range(1, 7):
        sample_freq = summary["state_frequencies"].get(point, 0.0)
        theo_freq = theoretical[point]
        print(f"{point:<5} {sample_freq:.2%}       {theo_freq:.2%}")
    
    # 7. 验证6点是否出现（理论上应为0）—— 更稳健的数组判断
    eff = mh.get_effective_samples()
    has_six = bool((eff == 6).any())
    print(f"\n有效样本中是否包含6点: {'是' if has_six else '否'}")
