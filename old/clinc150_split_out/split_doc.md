# CLINC150 TF-OWSL 语义划分说明（自动生成）

- Dataset: `cmaldona/All-Generalization-OOD-CLINC150`
- OOS label name: `ood`（始终作为 unknown，不参与扩类）

## 1. 划分目标
我们将 150 个 in-scope intents 划分为 `Y0`（上线即支持）+ 若干 `Phase-k`（随时间逐步出现的新语义）。
划分单位选择 **domain**，理由：
- domain 对应真实系统中的“功能模块”；新模块上线→相关查询会持续出现，更符合时间一致性假设。
- 避免随机拆分 intent 造成语义不可解释、触发/延迟统计不稳定。

## 2. 划分结果（domain 级）
- `Y0` domains（4 个）：['auto_and_commute', 'banking', 'credit_cards', 'home']
- phases（6 个）：每个 phase 引入 1 个 domain

### 2.1 Y0
- intents 数：60

### 2.2 Phases
- Phase-1: domain=`kitchen_and_dining`, intents=15
- Phase-2: domain=`meta`, intents=15
- Phase-3: domain=`small_talk`, intents=15
- Phase-4: domain=`travel`, intents=15
- Phase-5: domain=`utility`, intents=15
- Phase-6: domain=`work`, intents=15

## 3. 数据统计（运行脚本得到的真实计数）
### train
- total=15200
- in_scope=15000
- oos=200
- in_scope_by_domain（节选前 10 项）：
  - auto_and_commute: 1500
  - banking: 1500
  - credit_cards: 1500
  - home: 1500
  - kitchen_and_dining: 1500
  - meta: 1500
  - small_talk: 1500
  - travel: 1500
  - utility: 1500
  - work: 1500

### validation
- total=3200
- in_scope=3100
- oos=100
- in_scope_by_domain（节选前 10 项）：
  - UNKNOWN_DOMAIN: 100
  - auto_and_commute: 300
  - banking: 300
  - credit_cards: 300
  - home: 300
  - kitchen_and_dining: 300
  - meta: 300
  - small_talk: 300
  - travel: 300
  - utility: 300

### test
- total=7900
- in_scope=6900
- oos=1000
- in_scope_by_domain（节选前 10 项）：
  - UNKNOWN_DOMAIN: 2400
  - auto_and_commute: 450
  - banking: 450
  - credit_cards: 450
  - home: 450
  - kitchen_and_dining: 450
  - meta: 450
  - small_talk: 450
  - travel: 450
  - utility: 450

## 4. 为什么这样划分（可复现实验的关键）
1) **事件强度足够**：每次引入一个 domain（通常约 15 intents），避免“新语义太弱导致触发不稳定”。
2) **OOS 始终存在**：OOS 从头到尾混入，且永不转正为新 intent，用来评估误扩类风险。
3) **可解释可复现**：domain 列表排序后确定 Y0 与 phase 顺序，保证不同机器/不同时间跑出的划分一致。
