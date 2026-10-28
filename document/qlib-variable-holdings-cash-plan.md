# Qlib 可变持仓与现金仓位：交易规则和实现计划

- 状态：设计完成，待实现
- 立案：2026-10-27
- 范围：`qlib-zh/` 的 A 股/港股 TopK 回测、practice 全周期回测、训练后推理与调仓建议
- 核心目标：把可配置的 `K` 从“必须持有 K 只”改为“最多持有 K 只”；由模型得分准入条件动态决定 TopK 中哪些股票不买
- 本文示例：`max_holdings = 50`，但 50 不是代码上限
- 本轮不做：不修改策略代码、不改变线上默认参数、不重跑模型或回测

## 1. 结论

建议采用“可配置最大槽位 + 动态模型准入 + 硬资格过滤 + 固定槽位预算”规则：

1. `max_holdings=K` 是正整数配置，表示组合的最大股票槽位数，也是仓位计算的分母；本文用 50 举例，但算法不写死 50。
2. 模型先产生全市场最终排名，截取原始 TopK；新买不从第 K+1 名以后补位。
3. TopK 每只股票再经过可配置的 `score_gate`。门槛判断基于模型预测对应的成本后预期超额收益和置信度，不配置固定“不买数量”。
4. 当日模型不买数量 `S_score(t)` 是派生结果：
   `S_score(t) = TopK 中未通过 score_gate 的股票数`。
5. 通过模型门槛的股票继续经过 ST、流动性、上市时长、风险否决等硬资格过滤。
6. 若最终选中 `N(t)` 只，股票目标总仓位按 `N(t)/K` 缩放，不把 N 只重新归一化为满仓。
7. 例如 K=50，某日 43 只通过模型门槛、其中 3 只硬过滤失败，则空仓账户买 40 只、股票仓位 80%；下一期若只有 27 只通过全部条件，则目标股票仓位降到 54%。
8. 门槛配置决定严格程度，未买数量由每期模型结果动态产生；提高门槛通常增加现金，降低门槛通常减少现金。
9. 买不进、整手不足、费用不足形成的余额继续留在现金，不再分摊给其他股票。
10. 硬风险退出不受普通换手上限限制；普通排名换仓仍受 `n_drop` 或等价参数限制。
11. 停牌、跌停等导致旧仓卖不出时，该仓位保留为“阻塞卖出”，占用持仓槽位，不提前买入替代股票。
12. “有效地没有合格股票”和“数据异常导致没有结果”必须是两种状态：
   - 有效空集：卖出所有可卖持仓，目标为 100% 现金。
   - 数据异常：不新买、不误清仓，冻结现有持仓并报警。

不建议只在现有 `TopkDropoutStrategy` 前过滤分数。标准 TopK 策略并不保证“每缺一个槽位就保留 `1/K` 现金”，而且现有自定义链路中仍有重新归一化、自动回填和空目标不清仓等行为。应增加统一的组合策略层，再由 Qlib 适配器负责下单。

## 2. 当前实现核对

### 2.1 仓库中不是只有一条 TopK 链路

当前至少有四套相关语义：

1. A/H 股基线：
   - `qlib-zh/cn/baseline.yaml`
   - `qlib-zh/hk/baseline.yaml`
   - `qlib-zh/hk/run_baseline.py`
   - 使用 Qlib `TopkDropoutStrategy(topk=50, n_drop=5)`。

2. 网页训练和完整回测主链：
   - `app.py`
   - `qlib-zh/train_runner.py`
   - `qlib-zh/scripts/practice/run_stage2_walk_forward.py`
   - 当前默认 `hold_num=20`，并强制使用 `master_analysis`，不是基线 YAML 中的标准 TopK。

3. stage3～stage6 研究链：
   - `stage3_first_screen.py`
   - `stage4_risk_eval.py`
   - `stage5_second_screen.py`
   - `stage6_final_result.py`
   - 默认数量又存在 20 和 5 两套值，且包含筛空后放宽、回退原始分数等行为。

4. 推理和调仓建议：
   - `qlib-zh/infer_runner.py`
   - `qlib-zh/strategy_b_analyze.py`
   - API 展示原始模型 Top20，Strategy B 默认推荐 Top5，尚未输出统一的目标权重和现金比例。

因此，本设计中的 `K` 是可配置最大槽位数，50 只是当前需求示例。不应在开发时简单地把所有 `20/5` 全局替换成 `50`。各入口要先统一组合规则，再由配置明确 `max_holdings` 和 `score_gate`；当日不买数量作为结果输出，不能作为固定输入。

### 2.2 现有行为与目标不一致的地方

`run_stage2_walk_forward.py` 已有 `PrecomputedWeightStrategy`，理论上能接收总和小于 1 的目标权重，但仍有以下问题：

- `_build_buffered_equal_weight_signal()` 用 `1 / 实际选中数` 分配权重。若只选中 30 只，仍然是 100% 股票仓位。
- 风险平价、stage5、stage6、大师分析和 fallback 分支大多会把剩余股票重新归一化到 1。
- `_RedistributingOrderGenWOInteract` 会把不可买股票的目标权重重新分给其他可交易股票，违背“买不成则留现金”。
- `generate_target_weight_position()` 对权重和小于 `0.99` 才保留现金；接近 1 的小额现金会被重新归一化吞掉。
- 空目标 `{}` 在当前订单生成器里直接返回空订单，并不等于卖出全部持仓。
- `master_analysis` 在候选数或因子数据少于 `hold_num` 时跳过整个日期。
- `risk_filter` 在安全股票不足时会把已否决股票补回。
- stage3 会在筛选过严时逐级关闭条件；stage5 和 stage6 也有回退原始候选的逻辑。
- `signal.csv` 和 `trade_signal.csv` 在现金缩放和 Master Veto 之前写出，不能代表最终目标。
- Qlib 返回的实际 `_positions_normal` 没有落盘；当前持仓数诊断统计的是目标信号行数，不是成交后的实际持仓。

这些行为说明“少于 K 只”偶尔已经会发生，但只是数据不足、不可交易或执行残差产生的偶发现象，不是可验证的交易规则。

### 2.3 模型分数不能直接作为绝对市场门槛

`run_stage2_walk_forward.py::_combine_model_predictions()` 会先对各模型分数做每日截面高斯排名，再进行加权。最终 `score` 主要表达当天股票间的相对顺序：

- `score > 0` 大致只说明位于当天截面的上半区。
- 每天总会存在排名靠前的股票，即使当天所有股票的预期收益都很差。
- 仅使用 `rank`、`rank_pct` 或截面分位数，不能判断市场整体是否值得满仓。

因此要区分两类条件：

1. 硬资格条件：ST、停牌、上市时长、流动性、价格/整手可买、风险否决等。它们可以直接造成合格股票不足，从而留现金。
2. 绝对质量门槛：预测的成本后超额收益、成功概率或经过 inner purged OOF 校准的置信度。它才能让策略在“全市场信号都弱”时主动增加现金。

硬资格过滤可以独立实现，但它不能完成“根据模型强弱动态决定不买数量”的核心目标。动态持仓模式启用前，绝对质量门槛及其校准产物必须先完成；校准只能使用预测日之前的 inner OOF 数据，不能在 outer test 或实盘未来数据上反向选阈值。

### 2.4 当前 walk-forward 不是每日重训

当前网页默认窗口是 `5year/1year/2year`。现有实现的真实语义是：

1. 每个 outer fold 训练一次模型。
2. 训练后一次性预测整个 test 段，`all_scores.csv` 保存测试段内所有交易日的分数。
3. 完整回测把各 fold 的整段预测拼接成逐日信号。
4. 测试段内虽然每天都有预测值，但模型 checkpoint 不变，不是每天重训。
5. `/api/qlib/infer` 显式使用 `--predict-only`，只加载最近已完成 fold 的 checkpoint；重新生成截至预测日前一日的 dataset 配置不等于调用 `fit()`。
6. predict-only 多模型路径还会因没有当次 validation IC 而改用等权 ensemble，与训练 fold 的权重存在不一致。

以 `walk_forward_end=2026-10-20`、训练 5 年、验证 1 年、测试 2 年为例，按交易日对齐后大致是：

- 训练样本日期：2018-10-22 ～ 2023-10-20。
- 验证样本日期：2023-10-23 ～ 2024-10-18。
- OOS 测试和逐日预测：2024-10-21 ～ 2026-10-20。

上述两年测试段共用同一个模型。2026-10-20 的输入特征是当日截面的最新特征，但模型参数仍来自在测试段开始前冻结的 checkpoint。

这适合作为“分块 OOS 回测”基线，但不能称为“单日滚动重训”。本计划的新模式必须把 outer test window 改为单个预测日，并在每个配置的重训日生成新的模型、校准器和预测。

## 3. 目标和非目标

### 3.1 目标

- 任意正常决策日的目标持仓数不超过可配置的 `max_holdings`。
- 通过可配置的模型得分准入条件动态减少买入数量，并按空缺槽位形成现金。
- 合格股票不足时，空缺槽位转成可解释的现金。
- 选股、仓位、执行三个步骤分离。
- 回测、训练后推理、API 和调仓建议使用同一套组合规则。
- 每只股票为什么入选、剔除、保留或卖出都可追踪。
- 目标持仓与实际成交持仓分别记录。
- strict daily 模式下每个预测日重新训练，并保存模型、校准器和数据截止日。
- 保持只做多、不使用杠杆。
- 不引入前视数据，T 日决策最早在下一个交易日执行。

### 3.2 非目标

- 不把可变持仓机制当作未经验证的市场择时。
- 不在第一版引入协方差优化、做空、杠杆或衍生品。
- 不在第一版接真实券商。
- 不更换 LightGBM 等模型算法、Alpha158 因子或单股大师评分；但会重构训练编排为单日 outer fold、inner OOF 校准和最终重训。
- 不把 warm-start 或加载旧 checkpoint 视为已经完成每日最终重训。
- 不保证因停牌、跌停等执行限制下的实际持仓始终立即小于等于配置 K；但必须禁止继续新增，并记录偏差。

## 4. 术语和状态

### 4.1 数量

- `max_holdings=K`：最大股票槽位数和仓位分母。K 可配置，50 只是示例。
- `master_candidate_pool_size=M`：在产生最终排名前，需要计算昂贵大师因子的预选池大小。由 `ceil(K × master_candidate_pool_multiplier)` 推导，并保证 `M >= K`。
- `raw_topk`：最终模型或大师排名的前 K 名，是新买股票的最外层边界。
- `gate_value`：用于准入判断的模型值，推荐为 inner purged OOF 校准后的成本后预期超额收益。
- `model_reject_count=S_score(t)`：当日 raw_topk 中未通过模型门槛的数量，是输出指标而非配置。
- `hard_reject_count=S_hard(t)`：通过模型门槛后又被硬资格过滤的数量。
- `selected_count=N(t)`：本次最终目标股票数，满足 `0 <= N(t) <= K`。
- `actual_holding_count`：成交后实际持有股票数。
- `blocked_exit_count`：已要求卖出但因不可交易仍留在账户中的股票数。
- `prediction_date=T`：当前单日 outer fold 唯一预测日期。
- `label_embargo=L`：标签表达式需要的最大未来交易日数。
- `train_label_end`：T 日决策时最后一个已获得完整标签的训练样本日期。
- `checkpoint_as_of`：模型、校准器和阈值共同对应的决策日期；strict daily 模式必须等于 T。

### 4.2 三种资格

同一只股票需要分别记录：

- `entry_eligible = score_entry_pass AND hard_entry_eligible`：是否允许新买。
- `hold_eligible = score_hold_pass AND hard_hold_eligible`：是否允许继续持有。为了减少抖动，模型持有门槛低于或等于模型入场门槛。
- `tradable_now`：执行日能否实际买卖。

例如，价格过高或一手买不起通常只禁止新买，不应自动强制卖出已有仓；跌停则可能允许继续持有但当日无法卖出。

### 4.3 决策状态

组合决策不得再用一个空 DataFrame 同时表达多种含义。至少需要：

- `VALID`：数据完整，已生成非空目标。
- `VALID_EMPTY`：数据完整，确实没有合格股票；目标为全现金。
- `DATA_ERROR`：行情、分数、校准器、阈值产物、因子或股票池数据不可用；冻结旧仓且不新买。
- `NO_REBALANCE`：当天不是调仓日；保持上一目标。

## 5. 交易规则

### 5.1 决策和成交时点

1. T 日收盘后只使用 T 日及以前可获得的数据计算资格、分数和目标持仓。
2. 订单最早在 T+1 的配置成交价执行。
3. 回测报告同时保存 `as_of_time` 和 `execution_time`。
4. 校准器、行业映射和历史股票池必须使用当时可获得的版本，不能使用今天的成分名单回填历史。

#### 5.1.1 单日 outer walk-forward

严格每日滚动模式下，每个预测日 T 都是一个独立 outer fold：

```text
历史已成熟标签
  → inner purged OOF 预测
  → 拟合/更新校准器和门槛
  → 使用全部已成熟标签最终重训
  → 只预测 T 一个交易日
  → T+1 执行
```

其中“已成熟标签”不是简单的 T-1。定义：

```text
L = 当前标签表达式引用的最大未来交易日数
latest_labeled_sample(T) = T 往前第 L 个交易日
```

- 普通 robust Alpha158 标签 `Ref($close,-2)/Ref($close,-1)-1` 的 L=2。
- multihorizon 默认最长标签使用到 T+22，L=22。
- 特征可以使用到决策日 T 收盘，但监督训练样本必须截止到 `latest_labeled_sample(T)`。
- train、inner OOF calibration 和 threshold selection 的末端都必须执行相同的 purge/embargo。

每个预测日的步骤：

1. 读取交易日历并计算标签 embargo L。
2. 截取最近 `train_window_years` 的已成熟训练样本。
3. 在该训练窗口内部生成 inner walk-forward OOF 预测；任何 OOF 行都不能由见过该行标签的模型产生。
4. 使用最近 `calibration_window_years` 的 OOF 预测拟合或更新校准器，并选择预先定义 profile 对应的门槛。
5. 冻结当日校准器、门槛和 ensemble 权重。
6. 使用截至 `latest_labeled_sample(T)` 的全部训练窗口最终重训模型。
7. 使用截至 T 收盘的特征只预测 T。
8. 保存以 T 为 as-of 的 checkpoint、校准器、阈值、训练边界和预测。
9. T+1 重新执行同一流程；不得直接复用 T 的 checkpoint 伪装成重训。

验证/校准区间不再永久从最终训练数据中扣除。它通过 inner OOF 保持样本外；校准完成后，最终模型可以重新使用这些日期的已成熟标签训练。

#### 5.1.2 2026-10-20 具体示例

假设：

- 信号在 2026-10-20 收盘后生成，2026-10-21 执行。
- 使用 robust 标签，L=2。
- 训练窗口 5 年。
- OOF 校准窗口 1 年。

则日期边界约为：

- 决策特征截止：2026-10-20。
- 最新已成熟训练样本：2026-10-16。该样本的 T+1/T+2 价格分别来自 10-19 和 10-20，决策时均已知。
- 最终训练样本窗口：约 2021-10-18 ～ 2026-10-16。
- inner OOF 校准样本窗口：约 2025-10-16 ～ 2026-10-16。
- 最终模型预测区间：仅 2026-10-20 一天。
- 目标执行日期：2026-10-21。

inner OOF 校准窗口中的每一行仍由更早数据训练的 inner model 预测。例如某个 2026-06-30 的 OOF 分数，不能来自包含 2026-06-30 标签的模型。校准器拟合后，再用 2021-10-18～2026-10-16 的全部已成熟样本最终重训模型。

2026-10-21 收盘后的下一次预测必须重新计算成熟标签截止日、更新 OOF/校准产物并生成新的 checkpoint。为降低成本可以复用特征缓存和已完成的 OOF 预测，但不能跳过最终重训。

若启用 multihorizon，最新已成熟样本改为 2026-10-20 往前第 22 个交易日，具体日期必须从 Qlib 交易日历计算，不能用自然日近似。

### 5.2 候选池和筛选顺序

建议顺序如下：

```text
历史时点股票池
  → 有效模型分数
  → 可选的大师候选池重排
  → 截取原始 TopK
  → 对 TopK 逐只计算动态模型准入条件
  → 对通过模型门槛的股票执行硬资格过滤
  → 持仓缓冲与换手约束
  → 最多选择 K 只
  → 计算目标总仓位和单票权重
  → 订单与实际成交
```

具体规则：

1. 先产生一个稳定的最终排名；分数相同时用股票代码升序保证结果可复现。
2. 从最终排名截取前 K 名，形成 `raw_topk`。
3. 对 raw_topk 每只股票计算 `gate_value`、`gate_confidence` 和 `score_gate_pass`。
4. `score_gate_pass=false` 的股票不买，对应槽位保留现金；其数量 `S_score(t)` 每期动态变化。
5. 通过模型门槛的股票继续计算硬资格；任一硬条件失败后保留空槽位。
6. 新买不得扫描第 K+1 名及以后补位，否则 TopK 内模型拒绝无法稳定映射成现金。
7. 当前旧仓可以因持仓缓冲暂时位于 TopK 之外，但会占用 K 个最大槽位之一，不能扩大总槽位。
8. 不得通过放宽门槛、恢复已否决股票或回退原始 TopK 来补满。
9. `master_candidate_pool_size` 只影响最终排名产生前的大师重排范围，不影响 K 和仓位公式。
10. 使用大师重排时必须满足 `M >= K`；例如 K=50 时可先重排 Top150，再从重排结果截取 Top50。
11. 若当日有效分数总数少于 K，`raw_topk` 取全部有效分数，未出现的槽位自然保留为现金，不报错补位。

### 5.3 可直接实现的选股与换手算法

#### 5.3.1 配置和派生值

每个调仓日先计算：

```text
K = max_holdings
D = max_regular_replacements
B = ceil(K × hold_rank_buffer_multiplier)
θ_entry = 模型入场门槛
θ_exit = 模型持有门槛
C_min = 最低校准置信度
```

约束：

```text
K >= 1
D >= 0
B >= K
θ_exit <= θ_entry
0 <= C_min <= 1
```

排名值和准入值必须分开：

- `rank_score`：现有每日截面高斯分数，只负责决定 TopK 顺序。
- `gate_value`：inner purged OOF 校准后的成本后预期超额收益，负责决定该股票是否值得占用槽位。
- `gate_confidence`：校准结果的置信度或样本可靠度。

#### 5.3.2 新买候选

1. 对所有有效分数按 `rank_score DESC, instrument ASC` 排序。
2. `raw_topk = ranked[0:K]`。
3. 对 raw_topk 每只股票读取模型原始预测分量，并通过当日 as-of 冻结的 inner OOF 校准器得到：

```text
expected_excess_return
gate_confidence
estimated_round_trip_cost
expected_net_excess
  = expected_excess_return
  - estimated_round_trip_cost
  - safety_margin
```

4. 模型入场条件：

```text
score_gate_pass =
  expected_net_excess >= θ_entry
  AND gate_confidence >= C_min
```

5. `score_gate_pass=false` 的股票标记为 `MODEL_SCORE_GATE_FAILED`，不买且不补位。
6. 对通过模型门槛的股票继续执行硬资格过滤。
7. `entry_pool` 为同时通过模型门槛和硬资格条件的 raw_topk 股票，保持原排名。
8. 不从 `rank > K` 的股票补位。

资格快照必须同时覆盖 raw_topk 和全部当前持仓：前者计算入场条件，后者计算较宽松的持有条件。当前持仓不在当日有效排名中时，其排名按正无穷处理，并由历史股票池、模型退出门槛和退市规则决定是否退出，不能因字典缺值直接异常。

各拒绝数量是当日派生指标：

```text
U(t)       = max(K - len(raw_topk), 0)
S_score(t) = count(raw_topk where score_gate_pass=false)
S_hard(t)  = count(raw_topk where score_gate_pass=true
                   and hard_entry_eligible=false)
```

拒绝原因按“有效分数不足 → 模型门槛 → 硬资格”顺序互斥归类，避免一只股票重复计数。

如果当前为空仓，则：

```text
selected = entry_pool
N = len(selected)
```

因此在没有旧仓和执行阻塞时，N 的精确公式为：

```text
N(t) = K - U(t) - S_score(t) - S_hard(t)
```

#### 5.3.3 旧仓处理

旧仓按以下优先级处理：

1. 硬退出：
   - 使用 `θ_exit <= θ_entry` 计算 `score_hold_pass`，形成迟滞区间。
   - `score_hold_pass=false` 或 `hard_hold_eligible=false` 的旧仓进入 `mandatory_sells`。
   - 风险否决、ST、退市或成本后预期超额收益跌破退出门槛属于硬退出。
   - 硬退出不占用普通替换额度 D。

2. 容量退出：
   - 去除硬退出后，若可保留旧仓数仍大于 K，按排名从差到好卖出超出 K 的部分。
   - 这是 K 下调产生的容量收缩，不得通过保留超额目标持仓绕过。

3. 排名缓冲：
   - 仍满足持有条件且 `rank <= B` 的旧仓可以暂时保留。
   - `rank > B` 的旧仓优先进入普通卖出候选。
   - 位于 `K < rank <= B` 的旧仓可保留，但当存在更高排名、满足入场条件且未持有的股票时，可以作为普通升级替换对象。

4. 普通替换：
   - 普通卖出候选按排名从差到好排序。
   - 每次最多选择 D 只。
   - 只有 `entry_pool` 中存在可买替代股票时，才为“升级排名”卖出仍可持有的旧仓。
   - 已跌出持仓缓冲 B 的旧仓即使没有替代股票也可以卖出，卖出后保留现金。

5. 空位补充：
   - `retained` 为计划卖出后仍保留的旧仓。
   - `vacancies = max(K - len(retained), 0)`。
   - 从 `entry_pool` 中排除 retained、现有阻塞买卖和重复股票，按排名取前 `vacancies` 只。
   - 空仓首次建仓和已有空位的补充不占用普通替换额度 D；D 只限制一进一出的普通换仓。

6. 最终目标：

```text
selected = retained + planned_buys
N = len(selected)
assert N <= K
```

#### 5.3.4 伪代码

```python
def build_target(scores, current_positions, eligibility, config):
    K = config.max_holdings
    D = config.max_regular_replacements
    B = ceil(K * config.hold_rank_buffer_multiplier)

    validate(K >= 1)
    validate(B >= K)
    validate(config.score_gate.exit_threshold <= config.score_gate.entry_threshold)

    if (
        scores.cross_section_status == "DATA_ERROR"
        or score_calibrator_unavailable(config.score_gate)
    ):
        return freeze_current_positions_no_buy()

    ranked = stable_sort(scores, by=["rank_score desc", "instrument asc"])
    raw_topk = ranked[:K]

    if missing_gate_inputs(set(raw_topk) | set(current_positions)):
        return freeze_current_positions_no_buy()

    gate_result = {
        stock: evaluate_score_gate(
            raw_prediction=scores[stock].raw_prediction,
            calibrator=config.score_gate.calibration_artifact,
            estimated_cost=estimate_round_trip_cost(stock),
            safety_margin=config.score_gate.safety_margin,
            entry_threshold=config.score_gate.entry_threshold,
            exit_threshold=config.score_gate.exit_threshold,
            min_confidence=config.score_gate.min_confidence,
        )
        for stock in set(raw_topk) | set(current_positions)
    }

    model_rejects = [
        stock for stock in raw_topk
        if not gate_result[stock].entry_pass
    ]
    hard_rejects = [
        stock for stock in raw_topk
        if gate_result[stock].entry_pass
        and not eligibility[stock].hard_entry_eligible
    ]

    entry_pool = [
        stock for stock in raw_topk
        if gate_result[stock].entry_pass
        and eligibility[stock].hard_entry_eligible
    ]

    current = set(current_positions)
    mandatory_sells = [
        stock for stock in current_positions
        if not gate_result[stock].hold_pass
        or not eligibility[stock].hard_hold_eligible
    ]

    retainable = current - set(mandatory_sells)

    # K 下调后先把目标容量降到 K；最差排名优先退出。
    capacity_sells = worst_ranked(
        retainable,
        count=max(len(retainable) - K, 0),
    )
    retainable -= set(capacity_sells)

    ranked_outside_buffer = [
        stock for stock in retainable
        if rank(stock) > B
    ]
    upgrade_candidates = [
        stock for stock in retainable
        if stock not in entry_pool
    ]

    available_entrants = [
        stock for stock in entry_pool
        if stock not in current
        and not has_blocking_order(stock)
    ]

    buffer_sells = worst_ranked(ranked_outside_buffer, count=D)
    remaining_drop_budget = D - len(buffer_sells)
    buffer_sell_set = set(buffer_sells)
    upgrade_pool = [
        stock for stock in upgrade_candidates
        if stock not in buffer_sell_set
    ]
    upgrade_sells = worst_ranked(
        upgrade_pool,
        count=min(remaining_drop_budget, len(available_entrants)),
    )
    regular_sells = buffer_sells + upgrade_sells

    planned_sells = set(mandatory_sells + capacity_sells + regular_sells)
    retained = stable_rank_sort(retainable - set(regular_sells))
    vacancies = max(K - len(retained), 0)
    planned_buys = first_ranked(
        [
            stock for stock in available_entrants
            if stock not in planned_sells
        ],
        count=vacancies,
    )

    selected = stable_rank_sort(list(retained) + planned_buys)
    N = len(selected)

    relative_weights = (
        allocation_method(selected) if N > 0 else empty_weights()
    )  # N > 0 时 sum = 1
    gross_target = (
        (1 - config.base_cash_ratio)
        * config.risk_scale
        * N / K
    )
    target_weights = relative_weights * gross_target

    return PortfolioDecision(
        decision_status="VALID_EMPTY" if N == 0 else "VALID",
        max_holdings=K,
        raw_topk=raw_topk,
        model_rejects=model_rejects,
        model_reject_count=len(model_rejects),
        hard_rejects=hard_rejects,
        hard_reject_count=len(hard_rejects),
        selected=selected,
        selected_count=N,
        target_weights=target_weights,
        target_cash_weight=1 - gross_target,
        planned_sells=worst_ranked(planned_sells),
    )
```

执行器必须以实际成交修正上述计划：

- 卖单未成交时，对应替代买单暂不执行。
- 阻塞卖出继续占用实际槽位。
- `planned_buys` 中不可买的权重转为现金，不转给其他股票。
- 若实际持仓已超过 K，只卖不买，直到回到 K 以内。

#### 5.3.5 决策日示例

假设没有旧仓、基础现金为 0、风险系数为 1：

- K=50，强信号日 49 只通过模型门槛、1 只硬过滤失败：N=48，股票 96%，现金 4%。
- K=50，中等信号日 43 只通过模型门槛、3 只硬过滤失败：N=40，股票 80%，现金 20%。
- K=50，弱信号日 18 只通过模型门槛、3 只硬过滤失败：N=15，股票 30%，现金 70%。
- K=30，22 只通过模型门槛、2 只硬过滤失败：N=20，股票约 66.67%，现金约 33.33%。
- K=50，所有股票都未通过模型门槛：N=0，目标为 100% 现金。

### 5.4 硬资格条件

第一版建议至少支持以下可配置规则：

- 当日存在有效模型分数。
- 股票属于该历史日期的目标股票池。
- 上市天数达到最低值，初始可沿用 60 个自然日或改成明确的交易日口径。
- 非 ST、非退市整理等禁止买入状态。
- 信号时点无已知停牌状态；执行时再由交易所模拟器检查真实可交易性。
- 过去一段时间成交额达到最低值。应使用绝对门槛，不只使用“截面后 20%”。
- 未触发硬风险否决。
- 一手股票加预计费用不超过单槽位预算。
- 使用未复权成交价检查价格和整手；复权价只用于模型和因子。

所有条件必须输出稳定的原因码，例如：

- `NO_SCORE`
- `OUT_OF_UNIVERSE`
- `NEWLY_LISTED`
- `ST_OR_DELISTING`
- `SUSPENDED`
- `LOW_LIQUIDITY`
- `RISK_VETO`
- `LOT_UNAFFORDABLE`
- `SCORE_GATE_FAILED`
- `MISSING_REQUIRED_DATA`

### 5.5 绝对质量门槛

推荐的最终模式是“inner purged OOF 校准后的成本后预期超额收益”：

1. 模型导出两套值：
   - `rank_score`：保留现有每日截面高斯组合，只用于排序。
   - `raw_prediction`：保存高斯排名前的各模型原始预测，用于跨日期的绝对质量校准。
2. 对每个单日 outer fold，在其历史训练窗口内部生成 inner OOF 预测，分别把每个模型的 `raw_prediction_m` 映射为未来超额收益 `f_m(raw_prediction_m)` 和置信度；不得使用模型对自身训练样本的拟合值做校准。
3. 映射第一版使用分箱单调映射或 isotonic calibration，要求简单、可复现，并避免高阶模型过拟合。
4. 使用已冻结的 ensemble 权重合成：

```text
expected_excess_return = Σ(weight_m × f_m(raw_prediction_m))
gate_confidence = Σ(weight_m × confidence_m)
```

其中 `confidence_m` 由校准分箱样本量和标准误差确定，并随校准器冻结；不能用模型分数绝对值临时冒充置信度。

5. 从校准预期收益中扣除预计双边交易成本和安全边际，得到 `expected_net_excess`。
6. 入场条件由 `entry_threshold` 和 `min_confidence` 决定；阈值越严格，当期 `S_score(t)` 通常越大。
7. 持有退出门槛 `exit_threshold` 低于或等于入场门槛，形成迟滞，减少临界值附近反复交易。
8. 校准器和阈值作为模型产物保存，推理和回测读取同一版本。
9. 校准器和阈值只能在当日 outer fold 的历史 inner OOF 数据上拟合、选择，对该日 outer test 锁定。

不建议把当前最终 `score` 直接与常数比较。它经过每日截面高斯排名，分布被人为稳定，`score > 0` 之类规则通常只会选出近似固定比例，无法可靠表达“今天整体信号弱”。若研究阶段临时启用 `relative_score_threshold`，报告必须标记它只控制相对持仓数量，不代表绝对收益判断。

兼容模式可以先只启用硬资格条件，但必须在报告中标记：

> 当前现金仓位只来自硬过滤或执行限制，不代表模型判断市场整体不值得持有。

若校准器缺失，生产默认行为为“不新增、保留仍满足持有条件的旧仓并报警”；研究回测可显式选择 `disabled` 兼容模式，使 TopK 全部通过模型门槛，但不得静默回退。

### 5.6 持仓缓冲和换手

将现有单一 `n_drop` 拆成更清晰的规则：

1. `max_regular_replacements`：普通排名原因每次最多替换多少只；基线可继续用 5。
2. `hold_rank_buffer_multiplier`：由 `ceil(K × multiplier)` 计算旧仓排名缓冲，避免 K 改变后仍写死绝对排名。
3. `mandatory_exit`：ST、退市、硬风险否决、绝对质量跌破退出门槛等强制退出。
4. 强制退出不受普通替换数量限制，但仍受停牌、跌停和 T+1 等执行约束。
5. 普通卖出和强制卖出产生的空位，只能由满足入场条件的股票补充。
6. 若卖出数量大于合格买入数量，差额转为现金。
7. 若卖单未成交，不提前买入对应替代股票。
8. 若实际持仓已超过 K，只允许卖出，不允许新增，直到回到上限以内。

现有 `FULL_BACKTEST_BUFFER_PCT=0.5` 表示可以把旧仓保留到全市场前 50%，范围过宽且与 K 无直接关系。新规则使用 K 的倍数，例如 `hold_rank_buffer_multiplier=2.0`；具体倍数通过验证期确定。

### 5.7 现金和权重

定义：

- `K`：最大持仓数。
- `N(t)`：最终选中股票数，`0 <= N(t) <= K`。
- `U(t)`：当日有效排名数量不足 K 造成的空槽位。
- `S_score(t)`：未通过动态模型门槛的数量。
- `S_hard(t)`：通过模型门槛后未通过硬资格条件的数量。
- `C_base`：固定基础现金比例，范围 `[0, 1)`。
- `R_t`：当期组合风险缩放系数，范围 `[0, 1]`。
- `p_i`：选中股票内部的相对权重，`Σp_i = 1`。

目标股票总仓位：

```text
gross_target = (1 - C_base) × R_t × N / K
```

单票目标权重：

```text
target_weight_i = gross_target × p_i
```

目标现金：

```text
target_cash = 1 - gross_target
```

目标现金可以拆解为：

```text
base_cash         = C_base
risk_cash         = (1 - C_base) × (1 - R_t)
dynamic_slot_cash = (1 - C_base) × R_t × (K - N) / K
```

三部分满足：

```text
target_cash =
  base_cash
  + risk_cash
  + dynamic_slot_cash
```

空仓或完全重建场景下，`K-N = U(t)+S_score(t)+S_hard(t)`。有旧仓缓冲时，旧仓可能占用原本空出的槽位，因此三类拒绝数量用于解释候选漏斗，最终现金仍以实际目标数 N 计算。执行失败产生的现金在实际账户中另行记录，不混入目标现金公式。

内部相对权重 `p_i` 可以按策略选择：

- 等权：`p_i = 1/N`。
- 模型分数权重。
- 风险平价权重。
- 大师分析现有权重。

本次改造只改变目标总仓位，不强制所有策略改成同一种内部权重。这样可以把“少持有就留现金”与“选中的股票如何分配”解耦，减少回归范围。

示例：

- K=50，N=50，基础现金 0，风险系数 1：股票 100%，现金 0%。
- K=50，模型门槛拒绝 7 只、硬过滤拒绝 3 只，N=40：股票 80%，现金 20%。
- K=50，模型门槛拒绝 20 只、硬过滤拒绝 3 只，N=27：股票 54%，现金 46%。
- K=50，N=37，基础现金 10%，风险系数 0.8：股票 `0.9×0.8×37/50=53.28%`，现金 46.72%。
- K=50，N=0 且状态为 `VALID_EMPTY`：股票 0%，现金 100%。

实现时 Qlib 策略的 `risk_degree` 应固定为 1.0，由上述目标权重唯一表达现金和风险缩放，避免再乘一次 0.95 或其他系数。

不得使用 `total < 0.99` 之类魔数判断是否保留现金。目标权重和是多少，就应严格保留多少。

### 5.8 不可交易和阻塞卖出

执行层规则：

1. 先生成卖单，再根据已释放的现金和槽位生成买单。
2. 买入涨停、停牌、整手不足或费用不足时，未使用资金留在现金。
3. 不把失败买单的权重重新分配给其他股票。
4. 旧仓不再合格但卖不出时：
   - 目标权重为 0。
   - 实际仓位保留并标记 `BLOCKED_SELL`。
   - 每个可执行日继续尝试卖出。
   - 占用一个实际持仓槽位。
   - 不提前买入替代股票。
5. 阻塞仓位可能使实际现金低于目标现金，这是执行偏差，不得通过其他股票超配来掩盖。
6. 实际股票仓位、实际现金、目标股票仓位和目标现金必须分别报告。

### 5.9 空目标和异常数据

需要显式对象表达决策，不能继续依赖空字典：

- `VALID_EMPTY`：
  - 目标股票列表为空。
  - 对所有可卖持仓发出卖单。
  - 不发买单。
- `DATA_ERROR`：
  - 不发新买单。
  - 不因缺数据清仓。
  - 保留旧目标或冻结当前实际持仓。
  - 输出错误来源并触发告警。
- `NO_REBALANCE`：
  - 不重新计算目标。
  - 延续上一次有效目标。

## 6. 建议配置契约

组合规则应由一个显式配置对象管理，不再让各脚本分别读取几十个环境变量。现有环境变量可以在迁移期映射到新对象。

建议配置结构：

```yaml
portfolio_policy:
  enabled: false
  max_holdings: 50
  allow_new_buys_outside_topk: false
  master_candidate_pool_multiplier: 3.0
  unfilled_slot_policy: cash

  rolling:
    mode: strict_single_day_oos
    signal_timing: close
    execution_lag_trading_days: 1
    prediction_frequency: every_trading_day
    retrain_frequency: every_prediction
    test_window_trading_days: 1
    train_window_years: 5
    calibration_window_years: 1
    inner_oof_step: month
    label_embargo_trading_days: auto_from_label
    final_refit: true
    allow_stale_checkpoint_days: 0
    warm_start: false
    cache_features: true
    cache_inner_oof_predictions: true

  allocation:
    method: preserve_strategy_allocator
    base_cash_ratio: 0.0
    risk_scale_source: none

  qualification:
    mode: strict_no_relaxation
    min_listing_days: 60
    exclude_st: true
    require_historical_universe: true
    require_liquidity: true
    require_lot_affordable: true
    hard_risk_veto: true
    data_error_policy: freeze_no_buy

  score_gate:
    mode: calibrated_net_excess
    profile: balanced
    rank_field: rank_score
    raw_prediction_fields: model_raw_predictions
    calibration_artifact: score_gate_calibrator.pkl
    thresholds_artifact: score_gate_thresholds.json
    entry_threshold_override: null
    exit_threshold_override: null
    min_confidence_override: null
    safety_margin_bps_override: null
    missing_artifact_policy: freeze_no_buy

  turnover:
    max_regular_replacements: 5
    hold_rank_buffer_multiplier: 2.0
    mandatory_exit_bypasses_limit: true
    buy_after_confirmed_sell: true

  execution:
    redistribute_failed_buy_weight: false
    retry_blocked_sell: true
    enforce_lot_size: true
    risk_degree: 1.0

  report:
    include_position_exposure: true
    include_rebalance_events: true
    include_target_actual_gap: true
    actual_position_frequency: daily
```

约束：

- `max_holdings` 必须是正整数；策略代码不把 50 写成上限。
- `strict_single_day_oos` 模式下 `test_window_trading_days=1`，每个预测日只能消费该日之前可获得的数据。
- `retrain_frequency=every_prediction` 时，每个预测日必须生成带 as-of 日期的新 checkpoint；不得加载旧 fold 后只更新 dataset。
- `label_embargo_trading_days` 必须从当前标签表达式的最大未来引用自动推导，不能写死为 0。
- `final_refit=true` 时，最终模型使用当日全部已成熟标签重新训练；校准器只能来自 inner OOF 预测。
- `allow_stale_checkpoint_days=0` 时，checkpoint as-of 与预测日不匹配必须失败，不得静默继续推理。
- `allow_new_buys_outside_topk` 在本方案中必须为 `false`。
- `master_candidate_pool_multiplier >= 1.0`；M 由 K 动态推导，不另写固定 150。
- `unfilled_slot_policy` 在本方案中必须是 `cash`。
- `0 <= base_cash_ratio < 1`。
- `0 <= risk_scale <= 1`。
- `hold_rank_buffer_multiplier >= 1.0`。
- 入场阈值不得低于退出阈值。
- `min_confidence` 必须在 `[0, 1]`。
- `profile` 只选择阈值产物中已在验证期冻结的配置，例如 `conservative/balanced/aggressive`，不得在测试期动态调参。
- 未加载校准器或阈值产物时禁止静默启用绝对质量门槛。

`enabled` 初始保持关闭，用于旧策略 A/B；完成验收后再决定默认值。

## 7. 目标模块边界

### 7.1 纯组合策略层

建议新增：

`qlib-zh/scripts/practice/portfolio_policy.py`

职责：

- 不直接调用 Qlib 回测器。
- 接收某个决策时点的分数、资格快照、当前实际持仓和配置。
- 处理 TopK 截断、动态模型准入、入场/持有资格、排名缓冲、换手上限和现金公式。
- 输出目标持仓快照和原因码。
- 设计为纯函数，便于单元测试。

建议核心输入：

- `as_of_time`
- `execution_time`
- `scores`
- `entry_eligibility`
- `hold_eligibility`
- `current_positions`
- `account_equity`
- `policy_config`

建议核心输出：

- `decision_status`
- `max_holdings`
- `raw_topk`
- `model_rejects`
- `model_reject_count`
- `hard_rejects`
- `hard_reject_count`
- `gate_value`
- `gate_confidence`
- `selected_count`
- `eligible_count`
- `blocked_exit_count`
- `target_gross_exposure`
- `target_cash_weight`
- `target_positions`
- `exclusions`
- `planned_actions`
- `diagnostics`

### 7.2 Qlib 执行适配层

建议新增或从现有 `PrecomputedWeightStrategy` 重构：

`qlib-zh/scripts/practice/variable_holding_strategy.py`

职责：

- 把目标持仓快照转换为 Qlib 订单。
- 区分全现金目标、数据错误和非调仓日。
- 先卖后买。
- 不重分配失败买单的权重。
- 跟踪阻塞卖单。
- 保持目标权重和小于 1。
- 把 Qlib 实际 position 展平并落盘。

标准 `TopkDropoutStrategy` 可以继续保留为旧基线，但新可变持仓模式不应依赖其自然残余现金。

### 7.3 资格提供层

现有模块改为只提供资格、分数和原因，不再自行补满或最终归一化：

- `stage3_first_screen.py`：提供股票池、流动性、上市时长和交易状态。
- `stage4_risk_eval.py`：提供风险标签和数据完整度。
- `stage2_master_strategy.py`：提供重排分数和风险否决，不直接决定满仓。
- `stage5_second_screen.py`：提供组合约束结果，不把不足数量重新归一化为 1。
- `strategy_b_analyze.py`：提供 veto 和原因，不再在因子缺失时回退原始 TopN。

## 8. 分阶段实现计划

### 阶段 0：冻结语义和配置

涉及：

- `document/qlib-variable-holdings-cash-plan.md`
- 后续新增统一配置文件或配置类

任务：

- 确认 `K` 是最大持仓数。
- 确认 K 可配置，50 只作为默认示例而非硬上限。
- 确认新模式为单日 outer OOS，每个预测日最终重训；旧两年分块预测仅作为兼容基线。
- 确认信号时点为 T 日收盘、执行为 T+1，并据此自动推导 label embargo。
- 确认固定“不买数量”不作为输入配置。
- 确认 `S_score(t)` 由每期 TopK 的模型准入结果动态产生。
- 确认 `score_gate` 使用哪项校准指标、入场门槛、退出门槛、最低置信度和安全边际。
- 确认新买不扫描第 K+1 名以后补位。
- 确认空槽位按 `1/K` 转现金。
- 确认普通换手上限和硬退出的优先级。
- 保留旧策略 feature flag，禁止直接覆盖历史结果。

完成标准：

- 回测、推理和 API 使用相同名词。
- 不再混用 `topk`、`hold_num`、`top_n` 表示候选数、展示数和实际持仓数。
- 所有入口都能明确输出 K、`S_score(t)`、`S_hard(t)` 和 N。

### 阶段 1：单日滚动重训、模型校准和门槛产物

主要修改：

- `qlib-zh/train_runner.py`
- `qlib-zh/infer_runner.py`
- `qlib-zh/scripts/practice/run_stage2_walk_forward.py`
- `qlib-zh/scripts/practice/gen_practice_yaml.py`
- `qlib-zh/scripts/practice/export_model_predict.py`
- `qlib-zh/scripts/practice/multi_horizon_label_handler.py`
- 模型训练产物目录

任务：

- 新增严格单日 outer fold 生成器：每个预测日的 test 只包含一个交易日。
- 根据标签表达式自动解析最大未来引用 L，并对训练、OOF 和校准边界执行 purge/embargo。
- 在现有 `rank_score` 之外保存高斯排名前的各模型 `raw_prediction`。
- 在训练窗口内部生成 inner walk-forward OOF 预测，不再依赖固定的一年 holdout validation。
- 仅使用截至预测日已成熟的 OOF 标签建立分数到未来超额收益的校准器。
- 将预计交易成本和安全边际纳入 `expected_net_excess`。
- 在 inner OOF 数据上选择 `entry_threshold`、`exit_threshold` 和 `min_confidence`。
- 校准完成后，用全部已成熟训练样本最终重训模型。
- 每个预测日保存新的 checkpoint、校准器、阈值、ensemble 权重、训练区间、标签截止日、样本量和误差。
- 特征和已完成 OOF 预测可以缓存；严格模式不得因缓存命中而跳过当日最终重训。

新增产物：

- `score_gate_calibrator.pkl`
- `score_gate_thresholds.json`
- `score_gate_validation.csv`
- `model_checkpoint.pkl`
- `rolling_asof_metadata.json`
- 根目录汇总 `rolling_training_daily.csv`
- 根目录汇总 `rolling_predictions.csv`

建议按 as-of 日期隔离：

```text
rolling_models/20261020/
  model_checkpoint.pkl
  score_gate_calibrator.pkl
  score_gate_thresholds.json
  ensemble_metadata.json
  rolling_asof_metadata.json
  prediction.csv
```

完成标准：

- 同一模型、同一日期的回测与推理得到相同 gate 值。
- 每个预测日只有一个交易日的 outer test 预测。
- 每个预测日都有与该日 as-of 匹配的新 checkpoint。
- 模型训练、inner OOF 和校准数据不包含预测日未来数据。
- `train_label_end <= prediction_date - label_embargo`。
- 门槛提高时，通过集合单调不增加。
- 缺少或版本不匹配的校准产物会进入 `DATA_ERROR`，不会静默全买或全卖。

### 阶段 2：纯组合策略和单元测试

新增：

- `qlib-zh/scripts/practice/portfolio_policy.py`
- `tests/qlib/test_portfolio_policy.py`

任务：

- 定义配置和输入/输出数据结构。
- 接入冻结的 score gate 接口。
- 实现资格漏斗、排名缓冲、普通换手和强制退出。
- 实现固定槽位现金公式。
- 实现 `VALID_EMPTY`、`DATA_ERROR`、`NO_REBALANCE`。
- 输出原因码。

必须覆盖：

- 参数 K=20/50/100。
- 模型门槛通过数为 0、1、K-1、K。
- 硬资格通过数为 0、1、K-1、K。
- `gate_value` 恰好位于入场/退出门槛上下边界。
- 置信度恰好位于最低置信度上下边界。
- 当前为空仓、部分持仓、满 K 只、已有超过 K 只。
- 旧仓满足持有门槛但不满足新买门槛。
- 硬风险退出和普通排名退出同时发生。
- 有效分数股票少于 K 时不补位，缺口转现金。
- 大师候选池倍数小于 1 时配置报错。
- 校准器缺失、版本不匹配和单股 gate 数据缺失。
- 基础现金和风险缩放叠加。

完成标准：

- 所有目标日满足 `N <= K`。
- N=K-1 时不会把剩余股票重新归一化为 100% 股票。
- `model_reject_count` 等于当日 TopK 中 gate 失败数量，不来自固定配置。
- 提高模型门槛后目标持仓数不会增加。
- 任一被否决股票不会因补满逻辑重新进入。

### 阶段 3：接入 practice 全周期回测

主要修改：

- `qlib-zh/scripts/practice/run_stage2_walk_forward.py`
- `qlib-zh/scripts/practice/stage2_master_strategy.py`
- 必要时新增 `variable_holding_strategy.py`

任务：

- 在所有筛选、price cap、veto 完成后，统一调用组合策略层。
- `master_analysis` 预选池至少扩大到 K，建议初始 `M=3K`；重排完成后仍只截取最终 TopK。
- 对最终 TopK 运行动态 score gate，输出 gate 值、置信度和拒绝原因，不扫描 K+1 以后补位。
- 完整回测只拼接各单日 outer fold 的当日预测，不再使用一个 checkpoint 产生整段两年 `all_scores.csv`。
- 候选或因子不足时输出少量目标，不再跳过整个日期。
- 删除风险股票不足时的回填。
- 修复零权重股票仍计入持仓数的问题。
- 移除 `0.99` 归一化魔数。
- 修复“空目标没有卖单”。
- 禁止不可买权重再分配。
- 将现金和风险缩放放进目标权重，Qlib `risk_degree=1.0`。
- 在最终变换后再写 `trade_signal.csv`。
- 展平 Qlib `_positions_normal`，按交易日生成实际持仓和现金快照。
- 在完整回测 HTML 和文本报告中加入持仓数量、股票仓位、现金比例及目标/实际偏差的时间序列。

新增输出：

- `portfolio_decisions.csv`
- `score_gate_decisions.csv`
- `target_positions.csv`
- `actual_positions.csv`
- `order_events.csv`
- `cash_exposure.csv`
- `position_exposure_daily.csv`
- `rebalance_summary.csv`

完成标准：

- 目标持仓、实际持仓和现金能逐日对账。
- `equity = cash + Σ(实际股数 × 未复权成交价)`。
- Qlib 实际 position 与报告持仓数一致。
- `full_backtest/overview.html` 可以直接查看目标/实际仓位变化和每次调仓原因。
- `report_of_backtest.txt` 包含仓位变化摘要，不只包含收益指标。

### 阶段 4：接入 A/H 股基线和微调回测

主要修改：

- `qlib-zh/hk/run_baseline.py`
- `qlib-zh/hk/run_rolling.py`
- `qlib-zh/cn/baseline.yaml`
- `qlib-zh/hk/baseline.yaml`
- `qlib-zh/finetune_alpha158.py`
- `qlib-zh/scripts/finetune_alpha158.py`

任务：

- 在 YAML 增加可变持仓配置。
- 保留标准 TopK 为对照组。
- 新模式使用共享组合策略和目标权重执行器。
- A/H 股分别复用已有整手和费用规则。
- 保存实际持仓，不再丢弃 `_positions`。
- 复用同一套 `position_exposure_daily.csv` 和仓位变化报告，避免基线与 practice 使用不同口径。
- 固定 Qlib 镜像版本或至少记录镜像 digest，避免第三方默认行为漂移。

完成标准：

- 同一份预测分数可在“旧 TopK”和“可变持仓”之间无损 A/B。
- 所有配置 K 下、N<K 的日期都满足既定现金公式，执行残差单独列示。

### 阶段 5：统一推理、API、前端和飞书

主要修改：

- `qlib-zh/infer_runner.py`
- `qlib-zh/strategy_b_analyze.py`
- `app.py`
- `static/index.html`
- `integration/lark_bot.py`
- `integration/lark_card.py`

任务：

- 推理返回“模型排名”和“组合目标”两个独立对象。
- API 不再固定 `top_n=20` 和 Docker 内 `HOLD_NUM=20`。
- 现有 `--predict-only` 旧 checkpoint 路径保留为兼容模式；严格模式若没有当日 checkpoint，先触发滚动重训或返回明确错误。
- API 增加 `max_holdings` 和可选 `score_gate_profile`。K 的运维安全上限由部署配置决定，不在策略代码中写死 50。
- 返回 K、`model_reject_count`、`hard_reject_count`、`selected_count`、`target_cash_weight`、gate 值、目标权重和排除原因。
- 有当前持仓时，调仓建议使用实际持仓和阻塞状态。
- Strategy B 返回空买入列表时不再回退原始股票。
- 前端和飞书明确显示“最大 K、模型门槛拒绝数、硬过滤拒绝数、最终选中 N、目标现金 X%”。

完成标准：

- 回测和推理对同一日期、同一输入产生一致目标。
- 推理结果返回 `checkpoint_as_of`、`train_label_end`、`label_embargo_days` 和 `retrained`。
- API 的 `stocks` 不再被误解为最终应满仓买入的原始 TopN。

### 阶段 6：影子运行和上线

任务：

- 记录校准区间、样本量、误差和漂移。
- 以 feature flag 先做影子回测和纸面组合。
- 对比不同门槛 profile 下的持仓数、现金比例、换手和样本外风险收益。
- 达到验收标准后，再决定是否默认启用以及网页采用哪组默认 K/score-gate profile。

完成标准：

- 在“全市场信号弱”时增加现金的结论来自冻结的校准门槛，不来自当天相对排名。
- 测试期和实盘推理不重新拟合阈值。

## 9. 验收与回归

### 9.1 强制不变量

- 目标持仓数永远不超过 K。
- 新买股票只来自原始 TopK，不来自 K+1 以后。
- `model_reject_count` 必须等于当日 TopK 中 `score_gate_pass=false` 的数量。
- 每只模型拒绝股票都有 gate 值、门槛、置信度和 `MODEL_SCORE_GATE_FAILED` 原因。
- 不买数量不得来自固定 count 配置。
- 所有新买股票都满足 `entry_eligible=true`。
- 所有保留股票都满足 `hold_eligible=true`，阻塞卖出除外。
- 被风险否决、行业拒绝或流动性拒绝的股票不会被回填。
- 权重被删除后不会分配给其余股票。
- 买单失败后资金留在现金。
- 有效空集对应明确清仓订单。
- 数据异常不触发误清仓。
- 阻塞卖出占用槽位，不提前买替代股票。
- 不使用未来数据。
- 不使用杠杆，费用后现金不得显著为负。

### 9.2 执行测试

至少覆盖：

- 新买股票涨停或停牌。
- 旧仓跌停或停牌。
- 卖单失败后禁止替代买入。
- 一手恰好可买、加费用后不可买。
- A 股 100 股、科创板起买数量、港股不同 lot size。
- 复权因子不为 1。
- 分红、拆股、上市不足、退市和无报价。
- T+1 和最短持有期。
- 普通换手上限 D=0/1/5/K。
- K=20/50/100。
- 模型 gate 通过数 0/1/K-1/K。
- 入场门槛、退出门槛和置信度边界。
- 连续两个预测日 T/T+1 分别生成新 checkpoint，训练标签截止日随成熟标签推进。
- robust 标签自动得到 L=2，multihorizon 默认自动得到 L=22。
- inner OOF 任一预测行都不出现在对应 inner model 的训练标签中。
- 特征缓存和 OOF 缓存命中时仍执行当日最终重训。
- strict daily 模式拒绝 checkpoint as-of 早于预测日的推理。

### 9.3 A/B 回测

固定三组基准：

1. 兼容基准：
   - 所有股票均合格。
   - `score_gate.mode=disabled`。
   - N=K。
   - 无基础现金和风险缩放。
   - 新旧策略的选股和目标总仓位应一致；差异必须可解释。

2. 动态模型门槛基准：
   - K=50，构造不同 gate 值，令最终 N=40、37、20、0。
   - 验证股票仓位分别为 80%、74%、40%、0%。
   - 提高门槛后 N 单调不增加。
   - 验证不存在补满和权重再分配。

3. 异常执行基准：
   - 旧仓跌停、新股涨停、停牌、整手不足。
   - 验证目标持仓与实际持仓分离、阻塞状态持续、现金可对账。

4. 滚动重训基准：
   - 构造连续 20 个交易日，每日只预测当日。
   - 每日 checkpoint、校准器和 prediction 都带独立 as-of。
   - 修改第 T 日之后的数据不能改变 T 日及之前的模型和预测。
   - 旧分块模式作为性能对照保留，但不得与 strict daily 结果混为同一策略。

### 9.4 新增报告指标

除现有年化、超额收益、Sharpe、IR、最大回撤、换手率和 IC/ICIR 外，必须增加：

- 每日候选数、入场合格数、目标持仓数和实际持仓数。
- 平均、最小、最大持仓数。
- 平均、P5、P95 和最大现金比例。
- 达到最大槽位 K 的日期比例。
- K、N、`U(t)`、`S_score(t)`、`S_hard(t)` 和动态槽位现金的逐日分解。
- gate 值、置信度、入场/退出阈值及阈值附近股票数量。
- 每日重训成功率、训练耗时、checkpoint 年龄和训练标签新鲜度。
- label embargo、inner OOF 覆盖率和校准样本数量。
- 每种排除原因的数量。
- 阻塞买入、阻塞卖出数量及持续天数。
- 目标和实际仓位偏差。
- 整手和费用残余现金。
- 资格违规数，必须为 0。
- 实际持仓超过 K 的日期数及原因。

先验收规则和会计恒等式，再比较收益。不能因为某次收益更高而接受补回风险股票、前视数据或现金对不上。

### 9.5 持仓数量与仓位变化报告设计

报告必须同时回答四个问题：

1. 模型当日最多允许多少只、最终目标多少只。
2. 目标股票仓位和目标现金各是多少。
3. 经过涨跌停、停牌、整手和费用后，实际持仓与目标差多少。
4. 持仓数量或仓位发生变化的原因是什么。

#### 9.5.1 时间与统计口径

- `decision_date`：使用已知数据形成决策的日期。
- `execution_date`：目标开始交给执行器的日期，通常是 decision_date 的下一交易日。
- 目标仓位从 `execution_date` 起生效，不能提前画在 decision_date，避免报告产生一日错位。
- 目标序列只在有效调仓事件更新：
  - `VALID`：切换到新目标。
  - `VALID_EMPTY`：目标股票仓位变为 0，目标现金变为 100%。
  - `NO_REBALANCE`：延续上一有效目标。
  - `DATA_ERROR`：延续上一有效目标并显示异常标记；没有上一目标时显示未知，不得当作 0 仓位。
- 实际持仓使用每个交易日收盘后的 Qlib position，不用目标信号行数替代。
- 报告必须标记 fold 起止、模型重训日、门槛 profile 变化日和数据异常日。

逐日核心公式：

```text
target_holding_count = count(target_weight_i > 0)
target_gross_exposure = Σ target_weight_i
target_cash_weight = 1 - target_gross_exposure

actual_position_value = Σ(shares_i × unadjusted_close_i)
actual_equity = actual_cash + actual_position_value
actual_gross_exposure = actual_position_value / actual_equity
actual_cash_weight = actual_cash / actual_equity

holding_count_gap = actual_holding_count - target_holding_count
exposure_gap = actual_gross_exposure - target_gross_exposure
```

只做多且不使用杠杆时，应满足：

```text
target_gross_exposure + target_cash_weight = 1
actual_gross_exposure + actual_cash_weight = 1
```

允许存在由费用和浮点精度造成的小误差，但误差阈值必须配置并在超限时报警。

#### 9.5.2 逐日仓位文件

新增 `position_exposure_daily.csv`，每个交易日一行，至少包含：

- `date`
- `fold_id`
- `decision_date`
- `effective_target_date`
- `decision_status`
- `score_gate_profile`
- `checkpoint_as_of`
- `retrained`
- `train_sample_start`
- `train_label_end`
- `calibration_start`
- `calibration_end`
- `label_embargo_days`
- `checkpoint_age_trading_days`
- `max_holdings`
- `raw_topk_count`
- `model_reject_count`
- `hard_reject_count`
- `target_holding_count`
- `actual_holding_count`
- `target_gross_exposure`
- `actual_gross_exposure`
- `target_cash_weight`
- `actual_cash_weight`
- `holding_count_gap`
- `exposure_gap`
- `blocked_buy_count`
- `blocked_sell_count`
- `daily_turnover`
- `transaction_cost`
- `equity`

该文件是报告图表的唯一逐日汇总数据源。HTML 不得重新从信号行数推导另一套仓位口径。

`target_positions.csv` 每个目标快照、每只股票一行，至少包含：

- decision/execution 日期
- 股票代码、最终排名
- rank score、gate value、gate confidence
- 入场/持有门槛
- 目标权重
- `keep/buy/sell/mandatory_sell`
- 入选或排除原因

`actual_positions.csv` 每个交易日、每只实际持仓一行，至少包含：

- 日期、股票代码
- 实际股数、未复权收盘价、市值
- 实际权重
- 对应目标权重
- 与目标权重偏差
- `NORMAL/BLOCKED_SELL/UNPLANNED_REMAINDER`

`rebalance_summary.csv` 每次调仓一行，至少包含：

- 调仓前/后目标持仓数
- 调仓前/后实际持仓数
- 目标/实际股票仓位和现金
- 模型拒绝、硬过滤、买入、卖出、保留数量
- 阻塞买卖数量
- 预计和实际换手
- 预计和实际费用
- 最大单票目标/实际权重
- 主要仓位变化原因

#### 9.5.3 HTML 报告页面

`full_backtest/overview.html` 增加“持仓与仓位变化”区域，按以下顺序展示：

根目录 `overview.html` 只展示关键仓位摘要和链接；完整逐日图表、事件表和单票轨迹放在 `full_backtest/overview.html`，避免在两个页面重复计算数据。

1. 仓位摘要：
   - 平均/最小/最大目标持仓数。
   - 平均/最小/最大实际持仓数。
   - 平均股票仓位、平均现金比例。
   - 满仓日、部分仓位日、全现金日比例。
   - 最大目标/实际仓位偏差。
   - 阻塞卖出最长持续天数。

2. 股票仓位与现金时间序列：
   - 实际股票仓位和实际现金使用逐日堆叠面积，合计为 100%。
   - 目标股票仓位和目标现金使用独立线条叠加。
   - 显示调仓日、模型重训日、`VALID_EMPTY` 和 `DATA_ERROR` 标记。
   - 净值曲线单独成图，不与仓位百分比共用纵轴。

3. 持仓数量时间序列：
   - K 作为上限基准线。
   - 目标持仓数 N 和实际持仓数分别展示。
   - 实际持仓超过 K 时突出显示，并关联阻塞原因。

4. TopK 准入漏斗：
   - 每个调仓日展示 raw TopK 数量。
   - 模型门槛拒绝数 `S_score(t)`。
   - 硬过滤拒绝数 `S_hard(t)`。
   - 最终目标持仓数。
   - 鼠标悬停或明细区域显示当期 gate profile 和阈值。

5. 目标与实际偏差：
   - 逐日显示 `exposure_gap` 和 `holding_count_gap`。
   - 同时显示 blocked buy/sell 数量。
   - 可定位“卖不出导致股票仓位高于目标”和“买不进导致现金高于目标”。

6. 调仓事件明细：
   - 每次调仓一行，可展开查看买入、卖出、保留和拒绝股票。
   - 显示调仓前后持仓数量和股票仓位。
   - 显示目标变化、实际成交、未成交及原因。
   - 支持按日期、fold、gate profile 和事件状态过滤。

7. 单票仓位轨迹：
   - 选择股票后展示目标权重、实际权重和持仓状态随时间的变化。
   - 用于核对缓冲保留、强制退出和阻塞卖出是否符合设计。

8. 模型训练新鲜度：
   - 每个预测日显示 checkpoint as-of、训练标签截止日和校准截止日。
   - strict daily 模式下 `checkpoint_age_trading_days` 必须为 0。
   - 标记当日是否完成重训、是否使用缓存及 label embargo 天数。
   - 若兼容模式复用旧 checkpoint，明确显示模型年龄，不能仍标记为“每日滚动重训”。

#### 9.5.4 文本报告

`report_of_backtest.txt` 除收益指标外，增加：

- 回测区间和 K。
- 使用的 score gate profile、校准器版本和阈值版本。
- 每日 checkpoint as-of、训练标签截止日、校准截止日和模型年龄。
- 目标/实际持仓数的平均、最小、最大值。
- 目标/实际股票仓位及现金比例统计。
- `S_score(t)`、`S_hard(t)` 的均值和极值。
- 部分仓位、全现金、数据异常的交易日数量。
- 最大目标/实际仓位偏差及发生日期。
- blocked buy/sell 总数、最长持续时间和影响最大的日期。
- 仓位变化最大的前十次调仓及主要原因。

#### 9.5.5 报告验收

- HTML 图表、文本摘要和 CSV 对同一指标给出相同结果。
- 图表第一天和最后一天与回测报告日期一致，不因 forward-fill 多出日期。
- `VALID_EMPTY` 必须显示目标 0% 股票、100% 现金。
- `DATA_ERROR` 必须显示延续目标和异常标记，不能画成清仓。
- 每个调仓日满足 `target_holding_count = target_positions.csv` 中正权重股票数。
- 每个交易日满足 `actual_holding_count = actual_positions.csv` 中正股数股票数。
- 每个交易日的实际持仓市值与现金能还原 equity。
- 目标和实际仓位偏差可以由订单事件完整解释。
- strict daily 模式下每个预测日都能关联到同日 as-of 的 checkpoint 和校准器。
- 任一训练/校准截止日晚于允许的 label embargo 边界时，报告生成失败并标记前视风险。
- 报告在 N=0、没有成交、全程满仓和存在阻塞卖出时均能正常生成。

## 10. 风险和处理

### 10.1 收益下降

增加现金可能降低牛市收益。这是规则的预期结果，不应通过偷偷补满规避。报告应同时给出：

- 策略净值。
- 现金收益。
- 与旧满仓策略的风险收益差异。
- 单位股票仓位的收益贡献。

### 10.2 现金来自错误数据

若把因子服务失败当作“无合格股票”，会错误清仓。因此决策状态和数据完整度是上线前置条件。

### 10.3 分数校准漂移

模型重训、股票池变化和市场制度变化都会令分数到收益的映射漂移。校准器需要版本、训练区间和监控，不能写死一个长期阈值。

### 10.4 目标与实际持仓偏离

涨跌停、停牌、整手、费用和 T+1 都会造成偏离。报告必须以 Qlib 实际 position 为准，不能继续用信号行数代替实际持仓数。

### 10.5 多条入口再次漂移

主训练、基线、微调、推理和 Strategy B 目前各有参数。共享组合策略层是本方案的关键；只修改某一个脚本会很快再次出现 5/20/50 和两套 buffer 的漂移。

### 10.6 每日重训成本与可复现性

五年窗口每日全量重训和 inner OOF 的计算成本显著高于当前分块模式。优化顺序必须是：

1. 先实现无前视、逐日最终重训的正确基线。
2. 再缓存 Alpha158 特征、成熟标签、inner OOF 预测和校准分箱。
3. 允许并行计算互不依赖的 inner folds。
4. warm-start 只能作为显式实验模式；必须与从头重训做结果对照，不能默认认为两者等价。
5. 若因资源限制改为按周或按调仓日重训，配置和报告必须准确显示真实频率及 checkpoint 年龄，不能继续称为每日重训。

任何缓存键至少包含数据版本、股票池版本、标签配置、训练窗口、as-of 日期、模型参数和代码版本。缓存命中不得绕过 as-of 与 embargo 校验。

## 11. 开发前默认决策

若没有额外业务要求，开发时采用以下默认决策：

1. `max_holdings=K` 可配置；50 只是建议初始值，不是硬上限。
2. 新模式默认 `strict_single_day_oos`，每个预测日最终重训并只预测该日。
3. 标签 embargo 从表达式自动计算；校准使用 inner purged OOF，不永久空出最近一年。
4. 每日校准完成后，用截至当日已成熟标签的完整训练窗口最终重训。
5. 不配置固定跳过数量；当日 `S_score(t)` 完全由 score gate 结果推导。
6. `rank_score` 只负责排序，准入使用 OOF 校准后的成本后预期超额收益。
7. 第一版 score gate 同时检查 `expected_net_excess` 和 `gate_confidence`。
8. 入场/退出门槛来自冻结的当日 as-of 产物，退出门槛不高于入场门槛。
9. 新买只允许来自原始 TopK，不扫描 K+1 以后补位。
10. 大师重排预选池默认 `M=3K` 且强制 M>=K。
11. 空缺槽位保留现金，不向剩余股票再分配。
12. 普通换手上限沿用 5；硬退出和 K 容量收缩绕过该上限。
13. 旧仓使用 K 倍数排名缓冲，但只有在仍满足模型持有门槛和硬资格时才可保留。
14. 有效空集清仓；校准器或数据异常时冻结旧仓且不新买。
15. 不可交易卖单占用槽位，不提前买替代股票。
16. 目标现金只由组合策略层计算，Qlib `risk_degree=1.0`。
17. 先实现单日滚动校准、最终重训和 practice 全周期回测，再覆盖基线、微调和推理。
18. 保留旧分块模式和历史结果作为对照，完成 A/B 后再修改默认值。

## 12. 完成定义

- [ ] 存在统一的纯组合策略模块。
- [ ] K 可配置，并被统一解释为最大持仓槽位和仓位分母。
- [ ] 固定跳过数量不是配置项；动态拒绝数由模型 gate 逐期计算。
- [ ] 排名分数和准入分数分离，准入分数有无前视的校准产物。
- [ ] strict daily 模式每个预测日最终重训并只预测当日，不复用整段测试 checkpoint。
- [ ] label embargo 从标签表达式自动推导，训练和 inner OOF 均通过 purged 边界检查。
- [ ] 每日 checkpoint、校准器、阈值和预测均有一致的 as-of 元数据。
- [ ] 新买不使用 K+1 以后股票补位。
- [ ] 任意日期满足 N<=K，目标股票仓位严格按 N/K 缩放。
- [ ] 所有自动放宽、风险回填和失败买单再分配均已关闭。
- [ ] 有效空集、数据错误和非调仓日有独立状态。
- [ ] 回测、推理和 API 使用同一规则。
- [ ] 目标持仓、订单、成交、实际持仓和现金逐日可追踪。
- [ ] HTML、文本和 CSV 均能展示目标/实际持仓数量、股票仓位、现金比例及偏差。
- [ ] 仓位变化报告与 Qlib 实际 position、订单事件和账户权益逐日对账。
- [ ] 单元测试、异常执行测试和旧策略 A/B 全部通过。
- [ ] 测试期无阈值重拟合和前视数据。
- [ ] 网页默认 K/score-gate profile 经过独立回测决策，不在策略代码中写死。
