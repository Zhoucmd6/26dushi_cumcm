# 补学指南

你现有的 LP、IP、随机规划、动态规划和多目标决策基础已经能覆盖本题的大部分优化骨架。真正需要补的是“预测如何进入决策”和“决策如何随新信息滚动更新”。

## 必须补学

### 1. 滚动时域优化 / 模型预测控制（MPC）

用于：问题 3，兼顾问题 4。

最低学习目标：理解状态、预测时域、控制时域、“只执行第一段再重算”、终端储能约束，以及为什么不能使用未来实际数据。

检索词：

- `model predictive control microgrid battery energy management`
- `rolling horizon optimization energy storage`
- `微网 储能 模型预测控制 滚动优化`

### 2. 时间序列的滚动回测与数据泄漏

用于：问题 2 的日前负荷/光伏预测，问题 4 的电价预测。

最低学习目标：季节性朴素预测、带 144 点日周期的 SARIMA、树模型的滞后/日历特征、训练窗口、rolling-origin backtesting、MAE/RMSE/nMAE、预测提前量，以及训练数据必须早于预测日。历史平均必须保留为基线；机器学习只作为候选，不能因为更复杂就预设更优。

检索词：

- `rolling origin evaluation time series forecasting`
- `day ahead load forecasting seasonal naive baseline`
- `SARIMA 10-minute load forecasting seasonal period 144`
- `LightGBM day ahead load solar forecasting lag features`
- `solar power forecast error metrics nMAE`
- `时间序列 滚动预测 回测 数据泄漏`

### 3. 场景生成与两阶段随机规划

用于：问题 2，进而扩展到问题 4。

你已经学过随机规划，需要补到能实际构造场景：主模型第一阶段的日前购电和名义充放电在所有场景中相同，第二阶段只有紧急购电和富余处置随场景变化；用负荷与光伏的整日联合残差块抽样保持时间相关性。还要会读分解形式 `min c^T x + E[h(x,omega)]`，理解 recourse function、recourse matrix 和“一个公共第一阶段 + 多个场景补救块”。如果让电池日内适应新信息，则问题升级为多阶段或滚动控制。

检索词：

- `two stage stochastic programming battery scheduling scenarios`
- `block bootstrap forecast error scenarios energy`
- `nonanticipativity constraints energy scheduling`
- `two stage stochastic programming decomposed form recourse function`
- `L-shaped method Benders decomposition stochastic programming`
- `两阶段随机规划 储能 调度 场景生成 非预见性`

### 4. 分段线性费用与正负偏差变量

用于：问题 3 的上调、下调和违约结算。

最低学习目标：会把 `max(x,0)` 拆成非负正偏差和负偏差变量，知道何时不需要二进制变量，以及如何核对费用正负号。

检索词：

- `piecewise linear cost positive negative deviation linear programming`
- `imbalance settlement linear optimization day ahead real time`
- `分段线性化 正负偏差变量 线性规划`

## 建议补学

### 5. 报童模型与分位数决策

问题 2 中，正常购电与 5 倍紧急购电构成典型的不对称短缺成本。报童模型能解释为什么日前计划应偏向较高的净负荷分位数，而不是只取均值。

检索词：`newsvendor quantile asymmetric underage overage cost`、`报童模型 分位数 不对称成本`。

### 6. 混合整数储能互斥

用于：严格禁止同一时段同时充放电。你已有 IP 基础，只需掌握 big-M/二进制开关对充放电模式的表达，并理解它会增加计算量。

检索词：`battery charge discharge mutual exclusivity MILP`、`储能 充放电互斥 混合整数规划`。

### 7. 信息价值与消融实验

用于：回答问题 3 是否需要更多预报时刻。核心是逐步增加预测更新时间，对比成本下降和紧急购电下降，而不是只比预测 RMSE。

检索词：`value of information forecast energy scheduling`、`forecast value economic dispatch ablation`、`预测信息价值 调度`。

## 进阶可选

### 8. 鲁棒优化或机会约束（当前第三问不学、不实施）

当历史场景很少或担心极端误差时，可用误差区间做最坏情形优化，或要求供电不足概率不超过给定阈值。它们适合作为随机规划的稳健性对照，不是必需主线。

检索词：`robust microgrid energy management renewable uncertainty`、`chance constrained battery scheduling`。

### 9. 电池退化成本

可把每次充放电吞吐量乘一个小的退化单价，防止模型为很小价差频繁循环。题面没有给退化参数，因此最多作为扩展和敏感性分析，不能偷偷加入主目标。

检索词：`battery degradation cost throughput energy scheduling`、`储能 退化成本 吞吐量模型`。

## 现在可以不学

- 深度学习、Transformer：数据规模和题目目标不要求，用简单基线与场景法更可解释。
- 强化学习：训练和验证成本高，难以证明约束始终满足。
- 完整交流最优潮流：题目没有网络拓扑、电压和线路参数。
- 网络流：可以帮助直观理解能量跨时段流动，但不是必要求解框架。
- 复杂随机动态规划：状态和场景会迅速膨胀，滚动随机 LP 已足够。

## 推荐学习顺序

1. 先用半天掌握储能状态方程、效率和 kW/kWh 换算。
2. 再学滚动回测、季节性朴素预测和误差指标。
3. 把已有随机规划知识扩展到场景生成与非预见性。
4. 学 MPC 的“更新 - 重算 - 只执行前段”。
5. 最后补分段结算、报童分位数和信息价值。

学到这一步就足以理解并检查本项目的主模型，不需要先掌握所有进阶方法。

## 第三问创新方案的专项补学

建议按以下顺序学，并对照 `docs/model-q3.tex` 中公式：

1. 分段凸函数与次梯度：推导不同结算价格下的不调整区间。最低要求是理解折点左右导数夹住0的最优条件；检索 `newsvendor adjustment costs no adjustment region subgradient`。这是已有LP与凸优化知识的延伸。
2. 条件预报校准：按发布时刻、提前量构造误差分布；检索 `forecast calibration lead time probabilistic solar forecasting`。最低要求是只使用目标时段已经结束的历史残差，并保留负荷与光伏的时间相关性。
3. 决策导向预测：读07号文献，理解按调度费用而非RMSE选择校准参数；检索 `decision focused forecasting predict then optimize right hand side uncertainty`。本题先做有限候选校准，不要求端到端梯度，也不直接套用成本向量预测的SPO+证明。
4. CVaR线性化：读08号文献，掌握阈值变量和正部辅助变量；检索 `Rockafellar Uryasev CVaR scenario linear programming`。检查均值费用与尾部风险之间的取舍。
5. 信息价值与事件触发：比较8种可用更新时间组合，区分预报发布、重优化与调电；检索 `value of information energy scheduling event triggered MPC`。理解免费信息在同一正确优化模型下的价值非负，不虚构固定交易费用。
6. 提前量条件尺度与概率评价：读10号第3.3—3.4节、11号第4节。掌握位置尺度、收缩、CRPS、覆盖率和区间宽度；检索 `lead time dependent forecast error scale`、`probabilistic forecast calibration CRPS interval score`。本题指数与R映射为候选假设，需要对照，不是真实概率。当前不学、不实施Tube MPC和DRO。

## 第四问专项补学

对应 `docs/model-q4.tex`，不需要先学强化学习：

1. 条件期望与协方差：弄清E[价格×缺口]何时能拆开；检索 `conditional expectation joint electricity price net load`。
2. 价格加权报童分位数：自己推导80%及不调整区间，明确不是有储能时的通用逐时规则；检索 `newsvendor weighted quantile asymmetric cost`。
3. 分段线性期望损失：理解H函数的折点及场景辅助变量，复用LP知识；检索 `piecewise linear expected shortage cost`。
4. 简单电价预测与滚动验证：读12号第4、5、7节，理解LEAR启发但不照搬大特征集；指数残差修正是候选假设。
5. 优化需要哪些分布信息：读13号第3.2—3.4节。区分同一时段价量依赖、跨时段依赖、期望目标与全日CVaR；费用好不等于概率预测完全正确。
6. LP净化证明：保持库存增量不变，减少同时充放电且不增加紧急量；先核对免费富余等条件，再与MILP比较。
7. 本次Q4-2增加依赖收缩：理解经验联合分布与同边际独立乘积的凸组合，能推出应急费用系数即可；检索“dependence shrinkage mixture distribution”。不必先学Copula，rho不是相关系数。
8. 本次Q4-3增加信息分组与下一节点前瞻：读14号II-A、II-D，理解当前行动共同、下一更新后同组行动共同，以及桥接库存不能按未来场景提前分叉；检索“scenario bundling nonanticipativity battery scheduling”。
9. 继续运行价值：给定原始承诺和下一节点库存，求各组后续最小费用，再按组概率加权。它解释为什么当前会多存或少存电，是分解建模思路；暂不要求学习SDDP或自行实现Benders。
