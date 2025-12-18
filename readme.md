# Metropolis-Hastings (MH) 算法实现

本仓库实现了Metropolis-Hastings (MH) 算法的两种提议分布变体：
- **UniformProposalMH**: 基于均匀提议分布的MH算法
- **GaussianMH**: 基于高斯提议分布的MH算法

## 📚 算法原理文档

**想要深入理解算法的数学原理？** 请查看：
### [📖 ALGORITHM_PRINCIPLES.md - 算法数学原理详解](ALGORITHM_PRINCIPLES.md)

该文档详细说明了：
- MH算法的理论背景和数学基础
- 核心数学公式推导（接受-拒绝准则、细致平衡条件）
- 马尔可夫链理论与遍历性
- 高斯提议分布和均匀提议分布的数学原理
- 参数调优的数学指导
- 收敛性分析与诊断方法

---

# UniformProposalMH 均匀提议分布MH算法

## 类说明
`UniformProposalMH` 继承自 `BaseMetropolisHastings`，实现了基于均匀提议分布的Metropolis-Hastings算法。

### 特点
- 使用均匀分布作为对称提议分布
- 特别适合离散状态空间（如骰子点数、整数变量）
- 支持自定义步长控制探索范围
- 兼容连续状态空间

### 主要方法
1. `__init__()`: 初始化算法参数
   - `target_log_prob`: 目标分布的对数概率函数
   - `step_size`: 均匀提议的步长范围
   - `param_dim`: 参数维度
   - `state_validator`: 状态验证函数（可选）

2. `proposal_generate()`: 生成候选状态
   - 在当前状态±step_size范围内均匀采样
   - 对离散状态自动取整

3. `get_summary()`: 扩展摘要
   - 对离散状态计算各状态出现频率

### 使用示例
```python
# 骰子抽样示例
mh = UniformProposalMH(
    target_log_prob=dice_target_log_prob,
    step_size=1,
    param_dim=1,
    state_validator=dice_state_validator
)
mh.run()
summary = mh.get_summary()
```
### 参数建议
- 离散状态：step_size=1（相邻状态转移）
- 连续状态：step_size=0.1-1.0（根据目标分布尺度调整）

# GaussianMH 高斯提议分布MH算法

## 类说明
`GaussianMH` 继承自 `BaseMetropolisHastings`，实现了基于高斯提议分布的Metropolis-Hastings算法。

### 特点
- 使用高斯分布作为对称提议分布
- 适合连续状态空间
- 特别针对双峰分布优化
- 可通过proposal_sigma调整探索能力

### 主要方法
1. `__init__()`: 初始化算法参数
   - `proposal_sigma`: 高斯提议的标准差
   - 其他参数通过`**kwargs`传递（如n_iter, burn_in等）

2. `target_log_prob()`: 双峰目标分布
   - 默认实现为两个高斯分布的混合
   - 可重写此方法自定义目标分布

3. `proposal_generate()`: 生成候选状态
   - 当前状态+高斯噪声

### 使用示例
```python
# 双峰分布抽样示例
mh = GaussianMH(
    param_dim=1,
    n_iter=10000,
    proposal_sigma=1.5
)
mh.run()
print(mh.get_summary())
```
### 参数建议
- proposal_sigma: 1.0-2.0（根据目标分布峰间距调整）
- 对于更复杂分布，建议重写target_log_prob方法