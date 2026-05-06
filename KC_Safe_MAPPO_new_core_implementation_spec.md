# 安全约束多智能体强化学习方法实现说明（KC-Safe-MAPPO）

## 0. 文档定位

本文档用于在现有炼厂多装置操作调度项目中新增一个安全约束多智能体强化学习求解模块。当前项目已经包含基于 `config.py` 的运筹优化模型，例如 `refinery_gurobi_model.py` 和 `MODEL_IMPLEMENTATION.md`。本方法实现应与现有数学规划模型并行存在，而不是替代或覆盖已有代码。

本文档仅说明所提出的核心方法及其代码实现路径，不包含对比方法、消融实验或其他 baseline 的实现要求。

方法名称建议为：

**KC-Safe-MAPPO**  
**Knowledge-Constrained Safe Multi-Agent PPO for Refinery Multi-Unit Operation Scheduling**

中文名称：

**知识约束安全多智能体强化学习方法**

---

## 1. 与现有项目的关系

现有项目中的 `config.py` 是案例数据源，Gurobi 模型是确定性运筹优化求解器。新增的多智能体强化学习模块应复用同一份 `config.py` 数据，但应独立放在新的 `marl/` 目录下。

推荐关系如下：

```text
config.py
   ├── refinery_gurobi_model.py          # 已有：运筹优化模型
   └── marl/                             # 新增：安全约束多智能体强化学习方法
```

实现原则：

1. 不修改 `config.py` 的数据结构，除非确有必要；
2. 不修改或破坏已有 `refinery_gurobi_model.py`；
3. 新增 MARL 模块统一放在 `marl/` 目录；
4. 新增结果保存到 `results/marl/`；
5. Gurobi 结果和 MARL 结果分开管理；
6. `config.py` 只保存案例数据，不写算法超参数、奖励权重或训练设置；
7. MARL 需要的数据转换通过 `marl/utils/config_adapter.py` 完成。

---

## 2. 推荐项目结构

建议在现有项目中新增如下目录和文件：

```text
项目根目录/
├── config.py
├── refinery_gurobi_model.py
├── MODEL_IMPLEMENTATION.md
├── results/
│   ├── gurobi/
│   └── marl/
└── marl/
    ├── __init__.py
    ├── utils/
    │   ├── __init__.py
    │   └── config_adapter.py
    ├── envs/
    │   ├── __init__.py
    │   ├── refinery_env.py
    │   ├── safety_layer.py
    │   ├── routing.py
    │   ├── blending.py
    │   └── reward.py
    ├── algorithms/
    │   ├── __init__.py
    │   ├── networks.py
    │   ├── rollout_buffer.py
    │   └── kc_safe_mappo.py
    ├── configs/
    │   ├── algo_config.py
    │   └── train_config.py
    ├── experiments/
    │   ├── train_kc_safe_mappo.py
    │   └── evaluate_kc_safe_mappo.py
    └── tests/
        ├── test_env_reset.py
        ├── test_safety_layer.py
        ├── test_mass_balance.py
        └── test_training_smoke.py
```

---

## 3. 方法总体思想

炼厂多装置操作调度问题具有多装置耦合、库存动态、物流拓扑、调和过程、需求扰动和安全约束等特点。传统运筹优化模型能够清晰刻画装置能力、库存边界、物料平衡和需求满足等约束，但在动态扰动和在线重调度场景下可能存在求解时间较长的问题。

因此，本文将炼厂多装置操作调度问题转化为约束多智能体马尔可夫决策过程。每个核心装置作为一个智能体，根据局部观测输出负荷调整动作；环境根据装置动作计算物流、库存、调和和产品交付；安全层根据炼厂工艺知识对候选动作进行掩码和修正；训练阶段采用集中式 critic 评估全局状态价值，执行阶段各装置 actor 根据局部观测独立决策。

整体流程：

```text
config.py
   ↓
ConfigAdapter 读取案例数据
   ↓
RefinerySchedulingEnv 初始化炼厂环境
   ↓
各装置智能体根据局部观测输出候选动作
   ↓
SafetyLayer 进行动作掩码与动作修正
   ↓
RoutingModule 计算装置直连与缓冲路径物流
   ↓
BlendingModule 将调和组分转化为产品池库存
   ↓
RewardCalculator 计算奖励
   ↓
KC-Safe-MAPPO 更新 actor 和 centralized critic
```

---

## 4. 多智能体划分

本方法采用装置级智能体划分。7 个核心装置分别对应 7 个智能体：

```python
AGENTS = [
    "CDU1",
    "CDU2",
    "DFHC",
    "FCC1",
    "FCC2",
    "ROHU",
    "DHC",
]
```

各智能体含义：

| 智能体 | 对应装置 | 主要决策 |
|---|---|---|
| `CDU1` | 常减压装置 1 | 原油处理负荷 |
| `CDU2` | 常减压装置 2 | 原油处理负荷 |
| `DFHC` | 柴油加氢精制装置 | 柴油馏分处理负荷 |
| `FCC1` | 催化裂化装置 1 | FCC 进料处理负荷 |
| `FCC2` | 催化裂化装置 2 | FCC 进料处理负荷 |
| `ROHU` | 渣油加氢装置 | 渣油处理负荷 |
| `DHC` | 柴油加氢裂化装置 | 加氢裂化处理负荷 |

库存节点、缓冲节点、调和组分池和产品池不单独设置为智能体，由环境、路由模块和安全层统一处理。

---

## 5. ConfigAdapter 设计

### 5.1 文件位置

```text
marl/utils/config_adapter.py
```

### 5.2 作用

`config_adapter.py` 负责从现有 `config.py` 中读取数据，并转换为 MARL 环境可直接使用的数据结构。这样可以避免修改已有配置文件，也能确保 Gurobi 模型和 MARL 模型使用同一份案例数据。

### 5.3 建议接口

```python
class ConfigAdapter:
    def __init__(self, config_module):
        self.config = config_module

    def get_agents(self) -> list[str]:
        ...

    def get_units(self) -> dict:
        ...

    def get_inventory_nodes(self) -> dict:
        ...

    def get_product_pools(self) -> dict:
        ...

    def get_routes(self) -> dict:
        ...

    def get_yields(self) -> dict:
        ...

    def get_demands(self) -> dict:
        ...

    def get_prices(self) -> dict:
        ...

    def get_utility_costs(self) -> dict:
        ...

    def build_env_config(self) -> dict:
        ...
```

### 5.4 输出格式

`build_env_config()` 应输出如下结构：

```python
env_config = {
    "agents": [...],
    "time": {...},
    "units": {...},
    "inventory_nodes": {...},
    "product_pools": {...},
    "routes": {...},
    "yields": {...},
    "demands": {...},
    "prices": {...},
    "utility_costs": {...},
}
```

---

## 6. 环境设计

### 6.1 文件位置

```text
marl/envs/refinery_env.py
```

### 6.2 核心类

```python
class RefinerySchedulingEnv:
    def __init__(self, env_config: dict, seed: int | None = None):
        ...

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        ...

    def step(self, actions: dict[str, int]):
        ...

    def get_global_state(self) -> np.ndarray:
        ...

    def build_observation(self, agent_id: str) -> np.ndarray:
        ...
```

### 6.3 reset 输出

`reset()` 返回每个智能体的局部观测：

```python
obs = {
    "CDU1": obs_cdu1,
    "CDU2": obs_cdu2,
    "DFHC": obs_dfhc,
    "FCC1": obs_fcc1,
    "FCC2": obs_fcc2,
    "ROHU": obs_rohu,
    "DHC": obs_dhc,
}
```

### 6.4 step 输出

`step(actions)` 返回：

```python
next_obs, rewards, dones, infos
```

其中：

```python
rewards = {
    "CDU1": r_cdu1,
    "CDU2": r_cdu2,
    "DFHC": r_dfhc,
    "FCC1": r_fcc1,
    "FCC2": r_fcc2,
    "ROHU": r_rohu,
    "DHC": r_dhc,
}
```

```python
dones = {
    "CDU1": done,
    "CDU2": done,
    "DFHC": done,
    "FCC1": done,
    "FCC2": done,
    "ROHU": done,
    "DHC": done,
    "__all__": done,
}
```

`infos` 至少包含：

```python
infos = {
    "raw_actions": raw_actions,
    "safe_actions": safe_actions,
    "unit_loads": unit_loads,
    "flows": flows,
    "inventories": inventories,
    "product_delivery": product_delivery,
    "shortage": shortage,
    "global_reward": global_reward,
    "total_cost": total_cost,
    "energy_cost": energy_cost,
    "switch_cost": switch_cost,
    "shortage_cost": shortage_cost,
    "inventory_penalty": inventory_penalty,
    "violation_penalty": violation_penalty,
    "demand_satisfaction_rate": demand_satisfaction_rate,
}
```

---

## 7. 状态与观测设计

### 7.1 全局状态

全局状态用于 centralized critic，建议包括：

1. 当前时间比例；
2. 7 个装置当前负荷；
3. 7 个装置上一时段负荷；
4. 装置可用能力系数；
5. 主要库存节点库存；
6. 产品池库存；
7. 当前需求；
8. 上一时段动作；
9. 切换标志。

示例结构：

```python
global_state_dict = {
    "time": t / T,
    "unit_loads": {...},
    "last_unit_loads": {...},
    "unit_availability": {...},
    "inventories": {...},
    "product_pools": {...},
    "demands": {...},
    "last_actions": {...},
    "switch_flags": {...},
}
```

实现时展平为 `np.ndarray`。

### 7.2 局部观测

每个智能体局部观测包括：

1. 本装置当前负荷；
2. 本装置上一时段负荷；
3. 本装置可用能力；
4. 上游库存；
5. 下游库存或组分池库存；
6. 相关产品需求；
7. 邻近装置负荷；
8. 当前时间比例。

示例：

```python
def build_observation(self, agent_id: str) -> np.ndarray:
    if agent_id == "FCC1":
        obs_dict = {
            "self_load": ...,
            "last_self_load": ...,
            "availability": ...,
            "fcc_feed_buffer": ...,
            "gasoline_component_pool": ...,
            "diesel_jet_component_pool": ...,
            "lpg_component_pool": ...,
            "neighbor_FCC2_load": ...,
            "gasoline_demand": ...,
            "lpg_demand": ...,
            "time": ...,
        }
    ...
    return flatten(obs_dict)
```

---

## 8. 动作空间设计

### 8.1 第一版动作：离散负荷调整

第一版建议只实现离散负荷调整动作，不引入复杂运行模式，降低训练难度。

```python
LOAD_DELTA_ACTIONS = {
    0: -1,   # 下调一个负荷档位
    1: 0,    # 保持当前负荷档位
    2: +1,   # 上调一个负荷档位
}
```

负荷档位：

```python
LOAD_LEVELS = [0.0, 0.6, 0.8, 1.0]
```

对于每个装置，维护当前负荷档位索引：

```python
current_load_level_index[unit]
```

动作执行后更新索引：

```python
new_index = current_index + LOAD_DELTA_ACTIONS[action]
new_index = clip(new_index, 0, len(LOAD_LEVELS) - 1)
```

映射到物理负荷：

```python
physical_load = capacity_min + LOAD_LEVELS[new_index] * (capacity_max - capacity_min)
```

### 8.2 后续扩展

后续可以加入运行模式：

```python
action = (load_delta_action, mode_action)
```

但第一版不实现模式动作，仅预留接口。

---

## 9. 知识约束安全层

### 9.1 文件位置

```text
marl/envs/safety_layer.py
```

### 9.2 核心类

```python
class SafetyLayer:
    def __init__(self, env_config: dict):
        ...

    def get_action_mask(self, state: dict, agent_id: str) -> np.ndarray:
        ...

    def repair_joint_action(self, state: dict, raw_actions: dict[str, int]) -> dict[str, int]:
        ...

    def repair_load(self, unit: str, proposed_load: float, state: dict) -> float:
        ...
```

### 9.3 动作掩码

动作掩码用于屏蔽明显不可行动作：

```python
if upstream_inventory_low(unit):
    mask[INCREASE] = 0

if downstream_inventory_high(unit):
    mask[INCREASE] = 0

if current_load_at_min(unit):
    mask[DECREASE] = 0

if current_load_at_max(unit):
    mask[INCREASE] = 0
```

### 9.4 动作修正

动作修正用于把候选动作映射到安全负荷：

```python
safe_load = proposed_load
safe_load = min(safe_load, unit_effective_capacity_max)
safe_load = max(safe_load, unit_capacity_min_if_running)
safe_load = min(safe_load, upstream_available_material)
safe_load = min(safe_load, downstream_available_capacity)
```

如果上游没有物料或下游无法接收，应降负荷：

```python
if upstream_available_material <= 0 or downstream_available_capacity <= 0:
    safe_load = 0.0 if unit_can_shutdown else unit_min_load
```

### 9.5 安全约束优先级

若约束冲突，优先级为：

1. 库存不越界；
2. 装置能力不越界；
3. 物流拓扑可行；
4. 尽量减少需求缺口；
5. 尽量减少切换和能耗。

---

## 10. 物流路由模块

### 10.1 文件位置

```text
marl/envs/routing.py
```

### 10.2 核心类

```python
class FlowRouter:
    def __init__(self, env_config: dict):
        ...

    def route(self, state: dict, safe_loads: dict[str, float]) -> dict[tuple[str, str], float]:
        ...
```

### 10.3 路由原则

1. 装置之间允许直连；
2. buffer 仅代表真实缓冲或库存节点；
3. 优先使用直连主流程；
4. 多余或无法直接处理的物料进入对应缓冲池；
5. 缓冲池物料可在后续时段供给下游装置；
6. 未追踪流股进入 byproduct/recycle，不夸大为 loss；
7. 不允许违反拓扑连接关系。

### 10.4 路由实现逻辑

建议第一版采用规则路由：

```python
for unit in processing_order:
    required_feed = safe_loads[unit]
    upstream_nodes = get_upstream_nodes(unit)
    feed_alloc = allocate_feed(required_feed, upstream_nodes, inventories)

    outputs = apply_unit_yields(unit, feed_alloc)
    allocate_outputs(outputs, downstream_nodes, buffers, component_pools)
```

---

## 11. 调和模块

### 11.1 文件位置

```text
marl/envs/blending.py
```

### 11.2 核心类

```python
class BlendingModule:
    def __init__(self, env_config: dict):
        ...

    def blend(self, state: dict, demand: dict) -> dict:
        ...
```

### 11.3 基本逻辑

二次加工装置产物先进入调和组分池：

```python
gasoline_component_pool
diesel_jet_component_pool
lpg_component_pool
```

调和模块再根据产品需求和产品池可用容量生成最终产品：

```python
gasoline_blend = min(
    gasoline_component_inventory,
    gasoline_demand_remaining,
    gasoline_pool_available_capacity,
)

diesel_blend = min(
    diesel_component_inventory,
    diesel_jet_demand_remaining,
    diesel_pool_available_capacity,
)

lpg_blend = min(
    lpg_component_inventory,
    lpg_demand_remaining,
    lpg_pool_available_capacity,
)
```

第一版暂不实现复杂质量优化，但保留质量检查接口：

```python
def check_quality_constraints(self, product: str, recipe: dict) -> bool:
    ...
```

---

## 12. 奖励函数

### 12.1 文件位置

```text
marl/envs/reward.py
```

### 12.2 核心类

```python
class RewardCalculator:
    def __init__(self, env_config: dict, reward_config: dict):
        ...

    def compute(self, state: dict, transition_info: dict) -> tuple[dict[str, float], dict]:
        ...
```

### 12.3 全局奖励

全局奖励为综合运行成本的负值：

\[
R_t^{global}
=
-
(C_t^{energy}
+
C_t^{switch}
+
C_t^{shortage}
+
C_t^{inventory}
+
C_t^{violation})
\]

代码形式：

```python
global_reward = -(
    energy_cost
    + switch_cost
    + shortage_cost
    + inventory_penalty
    + violation_penalty
)
```

### 12.4 局部奖励

局部奖励用于缓解信用分配问题：

```python
local_reward[unit] = -(
    local_energy_cost
    + local_switch_cost
    + upstream_shortage_penalty
    + downstream_overflow_penalty
)
```

### 12.5 奖励融合

每个智能体最终奖励：

```python
agent_reward[unit] = alpha * global_reward + (1 - alpha) * local_reward[unit]
```

推荐：

```python
alpha = 0.7
```

---

## 13. KC-Safe-MAPPO 算法

### 13.1 文件位置

```text
marl/algorithms/kc_safe_mappo.py
```

### 13.2 网络结构

#### Actor

共享 actor：

```python
class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, agent_id_dim):
        ...

    def forward(self, obs, agent_id_onehot, action_mask=None):
        ...
```

输入：

```python
actor_input = concat(local_obs, agent_id_onehot)
```

输出：

```python
action_logits
```

动作掩码：

```python
masked_logits = action_logits + torch.log(action_mask + 1e-8)
```

#### Critic

集中式 critic：

```python
class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim):
        ...

    def forward(self, global_state):
        ...
```

critic 输入全局状态，输出状态价值：

```python
V(s_t)
```

### 13.3 Rollout Buffer

需要保存：

```python
obs
global_state
actions
action_masks
log_probs
rewards
dones
values
advantages
returns
```

### 13.4 训练流程

每个 rollout：

```python
for step in range(rollout_length):
    global_state = env.get_global_state()

    for agent in agents:
        obs_i = obs[agent]
        mask_i = safety_layer.get_action_mask(env.state, agent)
        action_i, log_prob_i = actor.sample(obs_i, agent_id, mask_i)

    next_obs, rewards, dones, infos = env.step(actions)

    buffer.store(...)
```

更新阶段：

1. 计算 GAE；
2. 计算 PPO clipped policy loss；
3. 计算 critic value loss；
4. 加入 entropy bonus；
5. 梯度裁剪；
6. 更新 actor 和 critic。

### 13.5 PPO 损失

策略损失：

\[
L^{clip} =
\mathbb{E}
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right]
\]

价值损失：

\[
L^V =
\mathbb{E}
\left[
(V_\phi(s_t)-\hat{R}_t)^2
\right]
\]

熵正则：

\[
L^{ent} =
\mathbb{E}[\mathcal{H}(\pi_\theta)]
\]

总损失：

\[
L =
-L^{clip}
+
c_v L^V
-
c_e L^{ent}
\]

### 13.6 推荐超参数

放入：

```text
marl/configs/algo_config.py
```

建议内容：

```python
ALGO_CONFIG = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_param": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "learning_rate": 3e-4,
    "max_grad_norm": 0.5,
    "rollout_length": 256,
    "ppo_epochs": 5,
    "mini_batch_size": 512,
}
```

---

## 14. 训练脚本

### 14.1 文件位置

```text
marl/experiments/train_kc_safe_mappo.py
```

### 14.2 功能

训练脚本应完成：

1. 读取现有 `config.py`；
2. 使用 `ConfigAdapter` 构建 `env_config`；
3. 初始化 `RefinerySchedulingEnv`；
4. 初始化 `KCSafeMAPPOTrainer`；
5. 进行训练；
6. 保存日志和模型。

### 14.3 命令示例

```bash
python marl/experiments/train_kc_safe_mappo.py \
  --config config.py \
  --total_steps 200000 \
  --seed 42 \
  --output_dir results/marl/
```

### 14.4 日志输出

保存：

```text
results/marl/training_log.csv
results/marl/kc_safe_mappo_checkpoint.pt
```

日志字段：

```python
{
    "episode": int,
    "total_reward": float,
    "total_cost": float,
    "energy_cost": float,
    "switch_cost": float,
    "shortage_cost": float,
    "inventory_penalty": float,
    "violation_penalty": float,
    "demand_satisfaction_rate": float,
    "inventory_violation_count": int,
    "unit_switch_count": int,
}
```

---

## 15. 最小测试要求

当前阶段只做所提方法的最小测试，不做对比实验。

### 15.1 环境 reset 测试

文件：

```text
marl/tests/test_env_reset.py
```

检查：

1. 返回 7 个智能体观测；
2. 库存初始化合理；
3. 装置负荷在能力范围内；
4. 时间步为 0。

### 15.2 安全层测试

文件：

```text
marl/tests/test_safety_layer.py
```

检查：

1. 上游库存不足时，上调动作被屏蔽；
2. 下游库存接近上限时，上调动作被屏蔽；
3. 修正后的负荷不超过装置能力；
4. 修正后的动作不导致明显库存越界。

### 15.3 物料平衡测试

文件：

```text
marl/tests/test_mass_balance.py
```

检查：

```python
new_inventory = old_inventory + inflow - outflow - delivery
```

允许小数值误差。

### 15.4 训练 smoke test

文件：

```text
marl/tests/test_training_smoke.py
```

运行少量训练步数，例如 1000 steps，检查：

1. 训练不报错；
2. loss 为有限值；
3. checkpoint 可保存；
4. 日志可输出。

---

## 16. 给 Codex 的直接提示词

```text
当前项目已经实现了基于 config.py 的炼厂多装置操作调度运筹优化模型，包括 refinery_gurobi_model.py 和 MODEL_IMPLEMENTATION.md。请不要修改或破坏已有 Gurobi 数学模型。

现在需要在同一项目中新增一个安全约束多智能体强化学习求解模块，实现 KC-Safe-MAPPO 方法。当前阶段只实现所提出的核心方法，不实现对比方法、消融实验或 baseline。

请按以下要求新增代码：

1. 保留 config.py 作为唯一案例数据源，不要在 config.py 中加入算法参数、奖励权重或训练建议；
2. 不要修改 refinery_gurobi_model.py 的已有逻辑；
3. 新增 marl/utils/config_adapter.py，用于把 config.py 中的装置、库存、收率、拓扑、需求等数据转换为 MARL 环境可用格式；
4. 新增 marl/envs/refinery_env.py，实现 RefinerySchedulingEnv，包括 reset、step、get_global_state、build_observation；
5. 新增 marl/envs/safety_layer.py，实现动作掩码和动作修正；
6. 新增 marl/envs/routing.py，实现装置直连与 buffer 路由；
7. 新增 marl/envs/blending.py，实现调和组分池到产品池的简化调和模块；
8. 新增 marl/envs/reward.py，实现全局奖励、局部奖励和局部—全局奖励融合；
9. 新增 marl/algorithms/networks.py，实现 SharedActor 和 CentralizedCritic；
10. 新增 marl/algorithms/rollout_buffer.py，实现 MAPPO rollout buffer；
11. 新增 marl/algorithms/kc_safe_mappo.py，实现 centralized critic + decentralized actor 的 KC-Safe-MAPPO；
12. 新增 marl/configs/algo_config.py，保存算法超参数；
13. 新增 marl/experiments/train_kc_safe_mappo.py，实现最小可运行训练脚本；
14. 新增 marl/tests/test_env_reset.py、test_safety_layer.py、test_mass_balance.py、test_training_smoke.py；
15. 所有 MARL 结果保存到 results/marl/，不要覆盖已有结果。

实现注意事项：
- 7 个装置智能体分别为 CDU1、CDU2、DFHC、FCC1、FCC2、ROHU、DHC；
- 智能体动作第一版只实现离散负荷调整；
- 装置之间允许直连，buffer 只表示真实库存/缓冲节点；
- 二次加工装置产物先进入调和组分池，再经调和模块进入最终产品池；
- 安全层必须防止明显库存越界、装置能力越界和拓扑不可行；
- reward 可参考数学模型目标函数，采用综合运行成本的负值；
- 训练采用 centralized critic 和 decentralized actor；
- actor 支持动作掩码；
- 所有随机性支持 seed；
- 训练日志保存为 CSV；
- 模型 checkpoint 可保存和加载；
- 保证新增代码与已有运筹优化模型互不冲突。
```

---

## 17. 当前实现范围总结

当前阶段只实现：

```text
config.py 读取
→ MARL 环境
→ 安全层
→ 路由
→ 调和
→ 奖励
→ KC-Safe-MAPPO
→ 最小训练脚本
→ 最小测试
```

当前阶段不实现：

```text
对比方法
消融实验
复杂质量调和优化
Gurobi-MARL 联合优化
完整实验结果分析
```

这些内容可在核心方法跑通后再逐步加入。
