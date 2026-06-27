# AFAC2026 赛题三：稀疏反馈下的自动化实验挑战 - 完整实验记录

## 1. 赛事概述

| 项目 | 内容 |
|------|------|
| 赛事名称 | AFAC2026挑战组-赛题三：稀疏反馈下的自动化实验挑战 |
| 主办方 | 蚂蚁集团 / 清华大学 |
| 奖金池 | ¥ 1,000,000 |
| A榜评测时间 | 2026/06/08 - 2026/07/21 |
| 评测指标 | 分类 Accuracy + 推荐 NDCG@10, 各占50% |
| 运行约束 | 单数据集 ≤ 2小时, 禁止并行, 禁止人工干预 |
| 允许API | Qwen3.5/3.6 系列, Qwen text-embedding-v4 |

### 赛题核心
构建 Agent 系统在有限预算、稀疏反馈、禁止并行约束下，自主完成多轮实验优化。

### 两个子任务
1. **产品分类（Task1）**: 图节点分类, 13,752节点, 767维特征, 10类, 图极度稀疏(平均度数2.04)
2. **产品推荐（Task2）**: 序列推荐, 50,000用户, 2,156物品, 测试用户序列极短(平均6.25)

---

## 2. 环境配置

### 硬件
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
- OS: Windows

### 软件
| 依赖 | 版本 |
|------|------|
| Python | 3.11.15 (conda: afac2026) |
| PyTorch | 2.5.1+cu121 |
| scipy | 1.17.1 |
| numpy | 2.4.4 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| openai | 2.43.0 |

### Conda 环境创建
```bash
conda create -n afac2026 python=3.11 -y
conda activate afac2026
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scipy pandas scikit-learn openai dashscope -i https://mirrors.aliyun.com/pypi/simple/
```

---

## 3. 提交历史与分数

| # | 时间 | 总分 | 分类 | 推荐 | 关键改进 | 效果 |
|---|------|------|------|------|----------|------|
| 1 | 06/25 11:27 | 0.5163 | 0.5631 | 0.4695 | 初始Agent(GraphSAGE+GRU4Rec, LLM决策3轮) | 基线 |
| 2 | 06/25 14:29 | 0.4884 | 0.7121 | 0.2646 | GCN+DropEdge+8集成, 推荐加流行度(失败) | 分类↑推荐↓ |
| 3 | 06/25 14:48 | 0.5913 | 0.7121 | 0.4705 | 修复推荐(关闭流行度, 回归CE) | 恢复 |
| 4 | 06/25 23:22 | 0.5903 | 0.7027 | 0.4780 | 15集成+序列增强+Agent轨迹日志 | 推荐↑ |
| 5 | 06/26 00:04 | 0.5947 | 0.7092 | 0.4801 | Agent V2(SASRec+CE, 诊断报告) | 推荐↑ |
| 6 | 06/26 01:45 | 0.5875 | 0.7201 | 0.4548 | 20集成+伪标签+SASRec滑动窗口 | 分类↑推荐↓ |
| 7 | 06/26 08:14 | 0.6001 | 0.7201 | 0.4800 | 伪标签+GRU4Rec回退(CE+随机截断) | 最佳组合 |
| 8 | 06/27 08:38 | 0.5983 | 0.7197 | 0.4768 | GRU4Rec+物品协同过滤(CF)混合 | CF无效 |
| **9** | **06/27 15:37** | **0.6066** | **0.7230** | **0.4902** | **GRU4Rec+用户特征融合(u_cat_01~08)** | **推荐↑新最佳** |
| 榜首 | - | 0.6391 | 0.7685 | 0.5097 | - | - |

### 趋势分析
- **分类**: 0.5631 → 0.7230 (提升0.16, 主要来自GCN+DropEdge+集成+伪标签)
- **推荐**: 0.4695 → 0.4902 (提升0.02, 主要来自序列增强+用户特征融合)
- **总分**: 0.5163 → 0.6066 (提升0.09)
- **距榜首**: 0.0325 (分类差0.046, 推荐差0.020)

---

## 4. 分类任务实验记录 (Task1)

### 4.1 提交 #1: GraphSAGE Baseline (测试 0.5631)

**Agent 3轮实验:**

| 轮次 | 模型 | hidden | layers | dropout | lr | norm | feat_norm | Val Acc | 策略 |
|------|------|--------|--------|---------|-----|------|-----------|---------|------|
| 1 | GraphSAGE | 128 | 2 | 0.3 | 0.005 | sym | l2 | 0.5740 | exploration |
| 2 | GraphSAGE | 256 | 2 | 0.5 | 0.01 | sym | l2 | 0.3751 | exploration (失败) |
| 3 | GraphSAGE | 64 | 1 | 0.1 | 0.001 | rw | standard | 0.4623 | exploitation |

**关键发现:**
- L2 特征归一化是关键（无归一化时 Acc=0.375, 有归一化时=0.574）
- 学习率 0.01 过高导致训练崩溃
- 对称归一化(sym)优于随机游走归一化(rw)

### 4.2 提交 #2-3: GCN + DropEdge + 集成 (测试 0.7121)

**关键改进:**
1. **GCN 替代 GraphSAGE**: 适合 transductive 任务, BatchNorm + 残差连接
2. **DropEdge**: 训练时随机丢弃20%边, 防止过平滑
3. **8模型集成**: 不同随机种子, 平均 softmax 输出
4. **标签传播后处理**: 基于特征相似度为稀疏节点补充标签(25%权重)
5. **余弦退火学习率**: 更精细的调度

**8模型集成结果:**

| 模型 | Seed | Val Acc | Epochs |
|------|------|---------|--------|
| 1 | 42 | 0.6939 | 250 |
| 2 | 52 | 0.7066 | 300 |
| 3 | 62 | 0.7012 | 250 |
| 4 | 72 | 0.7003 | 300 |
| 5 | 82 | 0.6885 | 150 |
| 6 | 92 | 0.6866 | 250 |
| 7 | 102 | 0.6921 | 200 |
| 8 | 112 | 0.6848 | 150 |
| **集成** | - | **0.6943** | - |

**配置:**
```json
{
    "model_type": "gcn", "hidden_dim": 256, "num_layers": 2,
    "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4,
    "epochs": 300, "patience": 50, "normalization": "sym",
    "feat_norm": "l2", "drop_edge_rate": 0.2, "label_prop_alpha": 0.25
}
```

**测试结果: 0.7121** (验证0.6943 → 测试0.7121, 测试>验证)

### 4.3 提交 #4: 15模型集成 + Agent 轨迹 (验证 0.6878)

**Agent 3轮 LLM 决策:**

| 轮次 | LLM决策要点 | Val Acc |
|------|------------|---------|
| 1 | baseline: hidden=256, layers=2, dropout=0.5, lr=0.01, label_prop=0.3, +度数特征 | 0.6893 |
| 2 | 增大集成容量: hidden=512, layers=3, dropout=0.7 | ~0.68 |
| 3 | 回归保守: hidden=128, layers=2, dropout=0.3 | ~0.69 |

**最终配置 (15模型集成):**
```json
{
    "model_type": "gcn", "hidden_dim": 256, "num_layers": 2,
    "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4,
    "epochs": 300, "patience": 50, "normalization": "sym",
    "feat_norm": "l2", "drop_edge_rate": 0.2, "label_prop_alpha": 0.3,
    "use_struct_feat": false
}
```

**15模型集成结果:** Val Acc = 0.6878 (个体范围 0.6794~0.7039)

---

## 5. 推荐任务实验记录 (Task2)

### 5.1 提交 #1: GRU4Rec Baseline (测试 0.4695)

**Agent 3轮实验:**

| 轮次 | 模型 | embed | hidden | lr | max_seq | Val NDCG | 策略 |
|------|------|--------|---------|-----|---------|----------|------|
| 1 | GRU4Rec | 64 | 128 | 0.001 | 50 | 0.5923 | exploration |
| 2 | SASRec | 128 | 128 | 0.005 | 20 | 0.5740 | exploration |
| 3 | GRU4Rec | 128 | 256 | 0.005 | 20 | 0.5815 | exploitation |

**关键发现:**
- GRU4Rec 优于 SASRec (稀疏新用户场景)
- 增大模型容量未必带来提升
- 验证-测试差距大 (0.59→0.47), 测试用户序列极短

### 5.2 提交 #2: 流行度混合 (测试 0.2646, 失败)

**改进尝试:**
- BPR 负采样损失替代 CE → NDCG 降至 0.4825
- 自适应流行度权重: 短序列用户用50-100%流行度 → 测试暴跌至 0.2646

**失败原因分析:**
- 流行度分数(0-1)与模型logits(-5~5)尺度不匹配
- 对短序列用户过度依赖流行度, 破坏了模型预测
- BPR损失收敛过快, 泛化能力下降

### 5.3 提交 #3: 回归纯模型 (测试 0.4705)

**修复:** 关闭流行度混合(popularity_weight=0)和去重(remove_seen=False), 回归CE损失

### 5.4 提交 #4: 序列增强 + Agent 轨迹 (验证 0.5911)

**关键改进: 序列增强**
- 训练时随机截断序列至5-15个item, 模拟测试集短序列分布
- 训练样本: 36,000 → 88,804 (增强52,804条)

**Agent 3轮 LLM 决策:**

| 轮次 | 配置要点 | Val NDCG | LLM分析 |
|------|---------|----------|---------|
| 1 | baseline: embed=64, hidden=128, lr=0.001, seq_aug=true | 0.5906 | "参考跨任务经验, 中等容量首轮最稳定" |
| 2 | 扩容: embed=128, hidden=256, layers=2, max_seq=100 | 0.5911 | "直击训练长序列vs测试短序列的分布偏移" |
| 3 | 回归: embed=64, hidden=128, max_seq=20, lr=0.005 | 0.5878 | "瓶颈不在表达能力, 而在短序列建模效率" |

**LLM跨任务经验迁移示例:**
> "参考跨任务经验：分类任务中中等容量（hidden_dim=128 vs 256）、适度正则化（dropout=0.2 vs 0.5）、适中学习率（lr=0.001 vs 0.01）在首轮即取得高稳定性"

---

## 6. Agent 系统设计

### 6.1 架构
```
┌──────────────────────────────────────────────────────┐
│                 AFACAgent 系统                        │
├──────────────────────────────────────────────────────┤
│  1. LLM 分析数据描述 + 历史实验记忆 + 跨任务经验       │
│  2. LLM 生成下一轮实验配置 (JSON)                      │
│  3. 运行实验: 训练模型 → 验证集评估                     │
│  4. 记录反馈: val_accuracy / val_ndcg                 │
│  5. 更新实验记忆, 提炼经验, 进入下一轮                  │
│  6. 预算耗尽后, 选择最佳模型生成最终预测               │
└──────────────────────────────────────────────────────┘
```

### 6.2 Agent 能力体现

| 赛题要求 | Agent 实现 |
|----------|-----------|
| 实验记忆维护 | 每轮记录config+指标+理由, 传递给下一轮LLM |
| 稀疏反馈提炼 | LLM分析val_acc/ndcg变化趋势, 识别瓶颈 |
| 探索/利用/停止 | LLM输出strategy字段, 指导探索或利用 |
| 跨任务经验迁移 | 分类经验传递给推荐(容量/正则/lr选择) |

### 6.3 轨迹日志

| 文件 | 内容 |
|------|------|
| `trajectory_B1.json` | 分类任务3轮: config+val_acc+ensemble_accs+rationale+strategy |
| `trajectory_B2.json` | 推荐任务3轮: config+val_ndcg+loss_type+rationale+strategy |

---

## 7. 模型技术细节

### 7.1 分类模型: GCN (改进版)

```python
class GCN(nn.Module):
    """GCN + BatchNorm + 残差连接"""
    # 每层: A_norm @ H → Linear → BatchNorm → ReLU → Dropout
    # 残差连接: 维度匹配时 h = h + h_in
```

**关键组件:**
- **DropEdge**: 训练时随机丢弃20%边, 防止过平滑
- **标签传播**: 基于特征余弦相似度, top-10邻居加权投票
- **集成**: 15个不同种子模型, softmax平均
- **特征归一化**: L2行归一化
- **图归一化**: 对称归一化 D^(-1/2)(A+I)D^(-1/2)

### 7.2 推荐模型: GRU4Rec

```python
class GRU4Rec(nn.Module):
    """GRU序列推荐"""
    # Item Embedding → GRU → Linear → 全物品打分
```

**关键组件:**
- **序列增强**: 随机截断训练序列至5-15个item, 36K→89K样本
- **CE损失**: 优于BPR损失(测试验证)
- **Pack Padded Sequence**: 高效处理变长序列

---

## 8. 项目文件结构

```
AFAC_3/
├── src/
│   ├── data_loader.py         # 数据加载 (CSR矩阵/CSV解析)
│   ├── models.py              # 模型定义 (GCN/GraphSAGE/GAT/GRU4Rec/SASRec)
│   ├── train_cls.py           # 原始分类训练 (Agent用)
│   ├── train_rec.py           # 原始推荐训练 (Agent用)
│   ├── train_cls_improved.py  # 改进分类 (DropEdge+集成+标签传播)
│   ├── train_rec_improved.py  # 改进推荐 (序列增强+BPR+流行度)
│   └── agent.py               # LLM Agent 决策模块
├── run.py                     # 原始Agent运行脚本
├── improved_run.py            # 改进版运行脚本 (最优配置直接运行)
├── agent_run.py               # Agent运行脚本 (LLM多轮决策+轨迹日志)
├── output/
│   ├── A1.csv                # 分类预测
│   ├── A2.csv                # 推荐预测
│   ├── trajectory_B1.json    # 分类轨迹日志
│   ├── trajectory_B2.json    # 推荐轨迹日志
│   ├── task1/round{1,2,3}/   # 每轮分类结果
│   └── task2/round{1,2,3}/   # 每轮推荐结果
├── experiment_log.md          # 本实验记录
├── competition_info.md       # 赛题信息
└── prediction.zip             # 提交文件
```

---

## 9. 运行说明

### 运行 Agent 系统 (LLM多轮决策)
```bash
conda activate afac2026
$env:PYTHONUTF8=1; python agent_run.py --budget 3 --task both --device cuda --n_ensemble 10
```

### 运行最优配置 (直接使用)
```bash
python improved_run.py --task both --device cuda --n_ensemble 15
```

### 仅运行分类/推荐
```bash
python improved_run.py --task cls --device cuda --n_ensemble 15
python improved_run.py --task rec --device cuda
```

---

## 10. 关键经验总结

### 分类任务
1. **L2特征归一化**是基础 (无归一化0.375 → 有归一化0.574)
2. **GCN优于GraphSAGE** (transductive任务, +0.12)
3. **DropEdge防止过平滑** (+0.02)
4. **集成学习稳定提升** (8模型: +0.03, 15模型: 边际递减)
5. **标签传播补充稀疏节点** (+0.01-0.02)
6. **学习率0.01最优**, 0.005太慢, 0.02不稳定

### 推荐任务
1. **CE损失优于BPR** (0.59 vs 0.48, 多次验证)
2. **流行度混合有害** (尺度不匹配, 0.47→0.26)
3. **序列增强有帮助** (模拟短序列分布, 36K→89K)
4. **GRU4Rec优于SASRec** (稀疏新用户场景, SASRec测试集更差)
5. **验证-测试差距大** (0.59→0.48), 核心问题是训练序列长(46)而测试短(6.25)
6. **用户特征融合有效** (u_cat_01~08, 冷启动用户+0.01, 提交#9: 0.4800→0.4902)
7. **物品协同过滤(CF)无效** (共现相似度尺度不匹配, 提交#8: 0.4800→0.4768)
8. **滑动窗口增强有害** (产生过多噪声短序列, 提交#6: 0.4801→0.4548)
9. **赛题要求使用用户特征和产品特征** (之前完全忽略user.csv/item.csv!)

### Agent 系统
1. **LLM能做出合理决策** (正确选择模型类型、识别瓶颈)
2. **跨任务经验迁移有效** (分类经验指导推荐配置)
3. **轨迹日志完整记录** (config+rationale+feedback+strategy)
4. **探索-利用平衡** (第1轮exploration, 后续exploitation)
5. **多维诊断反馈关键** (冷启动NDCG/孤立节点比例让LLM做更好决策)
6. **LLM会犯重复错误** (反复选择BPR, 需在提示词中注入经验教训)
7. **STOP_AND_ENSEMBLE机制** (预算控制, 体现赛题"停止决策"考点)

---

## 11. 后续提升方向

### 分类 (0.7230 → 目标0.75+)
- [ ] 更多集成模型 (25-30个)
- [ ] 更高标签传播权重 (0.5)
- [ ] 多轮伪标签迭代 (3轮以上, 置信度递减)
- [ ] GCNII / APPNP 调参 (APPNP alpha需要调高到0.5+)

### 推荐 (0.4902 → 目标0.52+)
- [ ] 物品特征融合 (item.csv: i_cat_01~03, i_bucket_01)
- [ ] GRU4Rec + SASRec 集成 (不同模型互补)
- [ ] Qwen text-embedding-v4 API (赛题允许, 增强物品表示)
- [ ] 短序列验证集 (匹配测试分布做模型选择)
- [ ] 两阶段: 检索(流行度+CF) → 排序(GRU4Rec+特征)
