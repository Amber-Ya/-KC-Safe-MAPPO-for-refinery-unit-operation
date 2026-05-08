# IEEE TII Baseline Suite

本目录集中管理投稿 IEEE TII 等期刊所需的对比方法。`marl/` 只保留 KC-Safe-MAPPO、环境和训练逻辑，所有 baseline 与消融实验入口放在这里。

## 已实现的可运行 Baselines

| 类别 | 方法 | 文件 |
|---|---|---|
| 基础策略 | Random Policy | `random_policy.py` |
| 启发式 | Rule-based dispatching | `rule_based.py` |
| 随机规划近似 | Scenario-based SAA | `stochastic_programming_saa.py` |
| 鲁棒优化近似 | CVaR robust planning | `robust_optimization_cvar.py` |
| 智能优化 | Genetic Algorithm | `genetic_algorithm.py` |
| 智能优化 | Particle Swarm Optimization | `particle_swarm_optimization.py` |
| 在线优化 | Rolling MPC-style planning | `rolling_mpc_milp.py` |
| RL/MARL 消融 | MAPPO/KC-Safe-MAPPO variants | `marl_variants.py` |

## 统一实验入口

普通不确定环境对比：

```bash
python baselines/run_comparison.py \
  --uncertainty_profile moderate \
  --train_steps 12000 \
  --eval_episodes 24 \
  --planner_candidates 80 \
  --planner_scenarios 8 \
  --mappo_profit_guard \
  --output_dir results/baselines/uncertainty_compare
```

TII 扩展 baseline 对比，包括 GA、PSO、rolling online baseline：

```bash
python baselines/run_tii_comparison.py \
  --uncertainty_profile moderate \
  --train_steps 12000 \
  --eval_episodes 24 \
  --planner_candidates 80 \
  --planner_scenarios 8 \
  --ga_population 20 \
  --ga_generations 6 \
  --pso_particles 20 \
  --pso_iterations 6 \
  --rolling_candidates 12 \
  --rolling_lookahead 4 \
  --mappo_profit_guard \
  --output_dir results/baselines/tii_moderate
```

RL/MARL 同族 baseline 与消融：

```bash
python baselines/marl_variants.py \
  --uncertainty_profile moderate \
  --train_steps 12000 \
  --eval_episodes 24 \
  --variants safe_mappo kc_safe_mappo_no_profit_guard kc_safe_mappo_profit_guard alpha_07 alpha_095 alpha_10 \
  --output_dir results/baselines/marl_variants
```

## 输出指标

所有统一评估入口输出：

- `mean_profit`
- `std_profit`
- `min_profit`
- `max_profit`
- `cvar10_profit`
- `mean_revenue`
- `mean_total_cost`
- `mean_inventory_violations`
- `mean_unit_switches`
- `mean_demand_satisfaction_rate`
- `average_decision_time_ms`

结果文件：

- `comparison_summary.json`
- `comparison_returns.csv`

## 投稿前仍建议升级的强基线

当前 SAA、CVaR robust、GA、PSO 和 rolling baseline 都是在 MARL 环境动力学上的 scenario/candidate planning 实现，已经可以用于实验流程和初稿对比。

投稿前若要进一步增强说服力，建议继续实现：

- Gurobi-based rolling-horizon MILP / MPC-MILP；
- Gurobi-based stochastic MILP SAA；
- Gurobi-based robust counterpart；
- Gurobi-based CVaR stochastic MILP。

这些需要在 `refinery_gurobi_model.py` 的 full-horizon deterministic MILP 基础上扩展多场景变量、非预期约束和滚动状态更新，不应与当前轻量级 baseline 混为一谈。

