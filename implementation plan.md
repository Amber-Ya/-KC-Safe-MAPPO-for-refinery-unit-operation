# KC-Safe-MAPPO 深度优化：逼近 Gurobi MILP

## 问题诊断

当前 gap = 20.1% (MARL profit 1,251K vs MILP 1,566K)。

**根因**：MARL 环境缺少 MILP 的 demand_max 约束，导致 MARL 无限制卖出产品。
Agent 学会了"全力开工"策略：加工 1643 吨原油（MILP 只用 964 吨），卖 5.14M 产品（MILP 只卖 3.95M）。
额外原油成本 (1.6M) > 额外收入 (1.2M)，净亏损 400K。

## 优化方案（3 层改进）

### 层1：环境对齐（最关键 — 预计 +200K~+300K profit）

#### [MODIFY] [blending.py](file:///Users/ya/Desktop/2026领域知识驱动的多装置协同调度/marl/envs/blending.py)
- 加入**每期需求上限（demand_max / horizon）**对 delivery 的约束
- 按 MILP 的 product_grades demand_max 除以 horizon 得到 per-period cap
- 这迫使 agent 学会只生产可卖出的量，减少无效原油消耗

#### [MODIFY] [refinery_env.py](file:///Users/ya/Desktop/2026领域知识驱动的多装置协同调度/marl/envs/refinery_env.py)  
- 在 `_current_demand()` 中传递 demand_max 信息给 blending

---

### 层2：训练管线增强（预计 +30K~+80K profit）

#### [MODIFY] [kc_safe_mappo.py](file:///Users/ya/Desktop/2026领域知识驱动的多装置协同调度/marl/algorithms/kc_safe_mappo.py)
- **Advantage 归一化**：batch-wise normalize advantages
- **Value clipping**：防止 critic 过拟合
- **Orthogonal 初始化**：提升 exploration quality

#### [MODIFY] [networks.py](file:///Users/ya/Desktop/2026领域知识驱动的多装置协同调度/marl/algorithms/networks.py)
- Orthogonal weight initialization (gain=sqrt(2) for hidden, 0.01 for output)
- LayerNorm 替代 Tanh（更稳定的梯度传播）

---

### 层3：Reward 信号精化（预计 +10K~+30K profit）

#### [MODIFY] [reward.py](file:///Users/ya/Desktop/2026领域知识驱动的多装置协同调度/marl/envs/reward.py)
- 降低 violation_cost 到 500（当前 35 violations × 1000 = 35K，结构性不可消除）
- 加入 **crude efficiency bonus**：奖励 revenue/crude_cost 比率

---

## 验证计划
1. 先运行 "oracle test"（固定 demand-capped delivery + 全满负荷）看 revenue ceiling
2. 训练 100K steps 并监控 profit 收敛曲线
3. 最终对比 MILP vs MARL (demand-aligned)