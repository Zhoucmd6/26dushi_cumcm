# 参考文献索引

本地已下载11篇公开全文。编号不代表重要性排名。第三问当前核心阅读为04、07、08、10、11；01、03、09中的鲁棒相关方法不进入本次模型。

Git协作版仅提供索引、公开来源和用途说明，不重新分发尚未逐篇核实再分发许可的PDF。已下载文件保留本地，团队成员可通过各条公开来源获取；本地文件链接在未下载时不会存在。

## 已下载全文

### 01 Robust Energy Management for Microgrids With High-Penetration Renewables

- 作者：Yu Zhang, Nikolaos Gatsis, Georgios B. Giannakis。
- 年份：2013。
- 来源：IEEE Transactions on Sustainable Energy；DOI `10.1109/TSTE.2013.2255135`；公开稿 https://arxiv.org/abs/1207.4831
- 本地文件：[01_Zhang_2013_Robust_Microgrid_Energy_Management.pdf](./01_Zhang_2013_Robust_Microgrid_Energy_Management.pdf)
- 对应本题：问题 2、4。
- 建议阅读：供需平衡、储能状态、可再生能源不确定集合和最坏情形交易成本。论文的方法比本题必需程度更复杂，重点借鉴不确定性怎样进入经济调度。

### 02 Model Predictive Control for Distributed Microgrid Battery Energy Storage Systems

- 作者：Thomas Morstyn, Branislav Hredzak, Ricardo P. Aguilera, Vassilios G. Agelidis。
- 年份：2017。
- 来源：arXiv `1702.04699`，https://arxiv.org/abs/1702.04699
- 本地文件：[02_Morstyn_2017_MPC_Distributed_Microgrid_BESS.pdf](./02_Morstyn_2017_MPC_Distributed_Microgrid_BESS.pdf)
- 对应本题：问题 3。
- 建议阅读：MPC 的状态、滚动时域和实时可计算性。该文包含网络潮流细节，本题没有网络参数，不需要照搬。

### 03 Real-time Operation Optimization of Microgrids with Battery Energy Storage System: A Tube-based Model Predictive Control Approach

- 作者：Cheng Lyu, Youwei Jia, Zhao Xu。
- 年份：2021。
- 来源：arXiv `2104.04819`，https://arxiv.org/abs/2104.04819
- 本地文件：[03_Lyu_2021_Tube_MPC_Microgrid_BESS.pdf](./03_Lyu_2021_Tube_MPC_Microgrid_BESS.pdf)
- 对应本题：问题 3、4。
- 建议阅读：预测误差下的滚动储能控制、终端 SoC 和稳健性。Tube MPC 可作为进阶方向，主模型不必一开始就采用。

### 04 Economic Evaluation of Stochastic Home Energy Management Systems in a Realistic Rolling Horizon Setting

- 作者：Julian Lemos-Vinasco, Amos Schledorn, S. Ali Pourmousavi, Daniela Guericke。
- 年份：2022。
- 来源：arXiv `2203.08639`，https://arxiv.org/abs/2203.08639
- 本地文件：[04_LemosVinasco_2022_Stochastic_HEMS_Rolling_Horizon.pdf](./04_LemosVinasco_2022_Stochastic_HEMS_Rolling_Horizon.pdf)
- 对应本题：问题 2、3、4。
- 建议优先阅读：PV、电池、随机场景和真实滚动回放被放在同一框架中，最接近本题“预测 + 决策 + 价格”的组合。特别注意其对完美信息、确定性和随机策略的对照。

### 05 Energy Storage Arbitrage in Two-settlement Markets: A Transformer-Based Approach

- 作者：Saud Alghumayjan, Jiajun Han, Ningkun Zheng, Ming Yi, Bolun Xu。
- 年份：2024。
- 来源：arXiv `2404.17683`，https://arxiv.org/abs/2404.17683
- 本地文件：[05_Alghumayjan_2024_Two_Settlement_Storage_Arbitrage.pdf](./05_Alghumayjan_2024_Two_Settlement_Storage_Arbitrage.pdf)
- 对应本题：问题 3、4。
- 建议阅读：日前与实时两阶段交易、偏差结算和储能套利。Transformer 不是本题必需内容，重点看结算结构和信息时间线。

### 06 Real-Time Energy Management for a Small Scale PV-Battery Microgrid: Modeling, Design, and Experimental Verification

- 作者：Mahmoud Elkazaz, Mark Sumner, David Thomas。
- 年份：2019。
- 来源：Energies 12(14), 2712；DOI `10.3390/en12142712`；https://www.mdpi.com/1996-1073/12/14/2712
- 本地文件：[06_Elkazaz_2019_Real_Time_PV_Battery_Microgrid.pdf](./06_Elkazaz_2019_Real_Time_PV_Battery_Microgrid.pdf)
- 对应本题：问题 1、3。
- 建议最先阅读：高层 MILP 先制定并网功率参考，实时预测控制器再跟踪，和本题从日计划到日内调整的层次最接近。

### 07 Smart “Predict, then Optimize”

- 作者：Adam N. Elmachtoub、Paul Grigas。
- 年份：2017年首次预印本；下载为2020年修订稿；期刊版2022年，Management Science 68(1), 9–26。
- DOI：10.1287/mnsc.2020.3922；公开稿：https://arxiv.org/abs/1710.08005
- 本地文件：[07_Elmachtoub_Grigas_Smart_Predict_Then_Optimize.pdf](./07_Elmachtoub_Grigas_Smart_Predict_Then_Optimize.pdf)，46页，已下载、解析并检查首页。
- 对应本题：问题2、3，决策费用驱动的预测选择或校准。
- 最值得看：引言中“预测误差”与“下游决策损失”的差别，以及SPO损失的定义。
- 适配限制：原始框架主要预测目标成本向量。本题负荷/光伏进入平衡约束右端，只借鉴决策导向思想；不能直接套用SPO+损失和一致性证明。建议先采用历史样本外调度费用校准少量参数。

### 08 Optimization of Conditional Value-at-Risk

- 作者：R. Tyrrell Rockafellar、Stanislav Uryasev。
- 年份：期刊版2000年，Journal of Risk 2(3), 21–41；下载为1999年9月5日作者稿。
- DOI：10.21314/jor.2000.038；作者公开稿：https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf
- 本地文件：[08_Rockafellar_Uryasev_CVaR.pdf](./08_Rockafellar_Uryasev_CVaR.pdf)，26页，已下载、解析并检查首页。
- 对应本题：问题3、4，紧急购电费用的尾部风险。
- 最值得看：CVaR辅助阈值及正部函数表示、有限场景线性化。金融应用的技术可迁移到电力成本，但金融参数不适用。
- 适配限制：CVaR增加了题面没有指定的风险偏好，必须保留风险中性基准并展示平均费用—尾部风险权衡。

### 09 A Distributionally Robust Model Predictive Control for Static and Dynamic Uncertainties in Smart Grids

- 作者：Qi Li、Ye Shi、Yuning Jiang、Yuanming Shi、Haoyu Wang、H. Vincent Poor。
- 年份：2024年预印本，arXiv:2403.16402。
- DOI：10.48550/arXiv.2403.16402；https://arxiv.org/abs/2403.16402
- 本地文件：[09_Li_2024_Distributionally_Robust_MPC.pdf](./09_Li_2024_Distributionally_Robust_MPC.pdf)，12页，已下载、解析并检查首页。
- 对应本题：问题3、4的进阶稳健性方案。
- 最值得看：Wasserstein分布集合、风险边界与可解重构；只借鉴误差分布不完全可信时的处理。
- 适配限制：原文还涉及网络与电动汽车等结构，本题不照搬。半径、支持集与缩放需重新定义，且应先证明普通场景模型确有不足。

### 10 Post-processing numerical weather prediction ensembles for probabilistic solar irradiance forecasting

- 作者：Benedikt Schulz、Mehrez El Ayari、Sebastian Lerch、Sándor Baran。
- 年份：2021；Solar Energy 220, 1016–1031。
- DOI：[10.1016/j.solener.2021.03.023](https://doi.org/10.1016/j.solener.2021.03.023)；[公开稿](https://arxiv.org/abs/2101.06717)。
- 本地文件：[10_Schulz_2021_Solar_Probabilistic_Postprocessing.pdf](./10_Schulz_2021_Solar_Probabilistic_Postprocessing.pdf)，32页，已下载、解析相关章节并核对元数据。
- 对应第三问：按预测提前量、发布时刻分别校准；过去日滚动训练；概率评分。
- 阅读重点：第3.3节（公开稿第9—10页）训练数据与分组，第3.4节CRPS。
- 适配限制：原文有天气集合，本题只有单条功率预报；不照搬其集合回归模型。文献没有给出本题的指数可信度公式，指数尺度和R映射均为本题待验证设计。

### 11 Probabilistic solar forecasting: Benchmarks, post-processing, verification

- 作者：Tilmann Gneiting、Sebastian Lerch、Benedikt Schulz。
- 年份：2023；Solar Energy 252, 72–80。
- DOI：[10.1016/j.solener.2022.12.054](https://doi.org/10.1016/j.solener.2022.12.054)；[大学机构库开放全文](https://publikationen.bibliothek.kit.edu/1000155949)。
- 本地文件：[11_Gneiting_2023_Probabilistic_Solar_Verification.pdf](./11_Gneiting_2023_Probabilistic_Solar_Verification.pdf)，9页，CC BY开放出版稿，已下载并核对相关章节。
- 对应第三问：概率预测、时间相关轨迹、校准与集中程度、覆盖率和区间评分。
- 阅读重点：第4节尤其4.4节（期刊第78页）；覆盖率并不能单独证明区间完全校准。
- 适配限制：本题不采用其中神经网络；历史联合块借鉴相关结构思想，不声称完整实现其经验Copula方法。R评分不等于文献定义的概率可靠性。

## 在线补充资料

以下资料未下载，逐项记录原因和阅读范围。此前NREL域名存在DNS解析失败，相关链接保留供浏览器直接访问。

### Event-triggered model predictive control for dynamic energy management of electric vehicles in microgrids

- 作者：Chuanshen Wu、Sufan Jiang、Shan Gao、Yu Liu、Haiteng Han。
- 年份：2022年，Journal of Cleaner Production 368, 133175。
- DOI及出版页：https://doi.org/10.1016/j.jclepro.2022.133175
- 状态：仅核对出版页摘要，未下载全文。
- 用途：问题3中“仅在必要时重新优化”的辅助方法。原文触发基于电动汽车状态预测误差，不能声称其直接证明本题的经济阈值。
- 适配限制：本题每天只有三个额外更新机会，没有单次调整固定费用。事件触发应检验计算量与费用的权衡，不能预先宣称优于完整信息优化的电费最优值。

第三问完整公式和逐段来源见 [model-q3.tex](../docs/model-q3.tex)，结构推导笔记见 [model-q3-innovation.md](../docs/model-q3-innovation.md)。

### Metrics for Evaluating the Accuracy of Solar Power Forecasting

- 作者：Siyuan Lu, Hendrik F. Hamann, Jie Zhang, Bri-Mathias Hodge, Anthony Florita, Venkat Banunarayanan。
- 来源：NREL，2014，https://www.nrel.gov/docs/fy14osti/60142.pdf
- 用途：问题 3 的光伏预测误差指标。重点看不要只用单一 RMSE，还要考虑预测提前量和经济价值。

### Solar Forecasting Best Practices Handbook

- 来源：NREL，2018，https://www.nrel.gov/docs/fy18osti/68886.pdf
- 用途：预测数据质量、基线、时间尺度和评估规范。

### Microgrid Energy Management and Methods for Managing Forecast Uncertainties

- 作者：Shanmugarajah Vinothine, Lidula N. Widanagama Arachchige, Athula D. Rajapakse, Roshani Kaluthanthrige。
- 年份：2022。
- 来源：Energies 15(22), 8525；DOI `10.3390/en15228525`；https://www.mdpi.com/1996-1073/15/22/8525
- 用途：快速比较随机规划、鲁棒优化、MPC 和蒙特卡洛场景法，适合作为方法地图。

## 最省时间的阅读顺序

1. 先读 06 的模型结构，理解微网、电池和滚动控制如何连在一起。
2. 再读 04，理解场景随机规划和真实滚动回放。
3. 读 05 的日前/实时结算部分，跳过 Transformer 细节。
4. 第三问优先读10的第3.3节与11的第4节，掌握提前量校准和概率评价。
5. 再读08的CVaR辅助阈值，07的决策导向思想。当前不要求学习01、03、09的鲁棒方法。
