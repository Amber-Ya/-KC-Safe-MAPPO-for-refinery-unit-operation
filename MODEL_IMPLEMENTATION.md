# 炼厂多装置操作调度 Gurobi 实现说明

实现文件：`refinery_gurobi_model.py`

## 运行方式

```bash
python refinery_gurobi_model.py --validate-only
python refinery_gurobi_model.py --solve --result-dir results
python refinery_gurobi_model.py --solve --quality-mode bilinear-crude --result-dir results_miqcp
```

注意使用 Anaconda `base` 环境里的 `python`。系统自带的 `/usr/bin/python3` 未安装 `gurobipy`，但 `/Users/ya/opt/anaconda3/bin/python` 已安装 Gurobi Python 接口。

## 模型类型

默认模型是 MILP，覆盖：

- 24 个调度时段；
- 6 类原油采购、库存与 CDU 进料；
- CDU1、CDU2、DFHC、FCC1、FCC2、ROHU、DHC 装置开停、能力上下限与模式互斥；
- CDU 产物在直连加工路径与缓冲罐路径之间分配；
- 中间缓冲节点、调和组分池、产品池库存平衡；
- 产品总需求上下限；
- 原油成本、公用工程成本、库存成本和产品销售收益。

可选 `--quality-mode bilinear-crude` 会加入 CDU 原油混合性质恒等式：

```text
q_mix[u,k,t] * charge[u,t] = sum_c crude_quality[c,k] * crude_feed[c,u,t]
```

这会把模型从 MILP 扩展为 Gurobi 可处理的非凸 MIQCP，并自动设置 `NonConvex = 2`。

## 主要决策变量

- `crude_buy[c,t]`：原油采购量；
- `crude_inv[c,t]`：分品种原油库存；
- `crude_feed[c,u,t]`：原油 `c` 到 CDU `u` 的进料；
- `unit_charge[u,t]`：装置处理量；
- `unit_on[u,t]`：装置开停二元变量；
- `mode_on[u,m,t]`：装置模式二元变量；
- `cdu_to_*`：CDU 产物到缓冲罐或直连装置的分流；
- `inventory[node,t]`：中间缓冲节点和调和组分池库存；
- `blend_to_product_pool[component_pool, product_pool, t]`：调和组分池到产品池流量；
- `product_sale[p,t]`：产品销售量。

## 目标函数

最大化利润：

```text
max product_revenue - crude_purchase_cost - utility_cost - inventory_holding_cost
```

其中公用工程成本由 `unit_utility_coefficients` 和 `utility_price` 逐装置线性计算。

## 数据限制说明

`config.py` 中包含产品质量规格 `product_quality_specs` 和详细调和组分清单 `blending_components`，但没有每个调和组分的性质、可用量、组分到产品的详细流量数据。因此当前实现不会强行构造缺数据的产品质量约束，而是在校验时给出警告。

若后续补充详细组分质量与供应数据，可在现有结构上扩展为按产品牌号的调和变量，并加入典型线性质量约束：

```text
sum_i quality[i,k] * blend[i,p,t] <= spec_max[p,k] * production[p,t]
sum_i quality[i,k] * blend[i,p,t] >= spec_min[p,k] * production[p,t]
```
