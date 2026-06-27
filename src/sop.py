"""
SOP (标准操作流程) 和知识库模块
基于9次提交的实验经验，为Agent提供决策辅助
"""


class KnowledgeBase:
    """9次提交的实验经验总结，注入Agent提示词防止重复错误"""

    CLASSIFICATION_LESSONS = """
## 分类任务经验库 (基于9次实验验证)
1. [验证✓] L2特征归一化是基础: 无归一化Acc=0.375 → 有归一化0.574
2. [验证✓] GCN优于SAGE和GAT: transductive任务GCN最佳, SAGE在KNN图上崩溃
3. [验证✓] DropEdge(0.2)始终有效: +0.03, 必须启用, 不可关闭
4. [验证✓] 集成15-20模型最优: 8模型→15模型提升明显, 15→20边际递减
5. [验证✓] 伪标签有效: 置信度>0.8的测试预测加入训练集, 二轮训练+0.01-0.02
6. [验证✓] 标签传播(0.3-0.4)有效: 基于特征相似度为稀疏节点补充标签
7. [验证✓] 学习率0.01最优: 0.005太慢, 0.02不稳定
8. [验证✗] KNN图增强失败: 高维空间(767维)引入噪声边, Acc从0.66暴跌至0.38
9. [验证✗] APPNP(alpha=0.1)失败: 传播过多导致崩溃, 需alpha>0.5
10. [验证✓] 对称归一化(sym)优于随机游走归一化(rw)
"""

    RECOMMENDATION_LESSONS = """
## 推荐任务经验库 (基于9次实验验证)
1. [验证✓] CE损失远优于BPR: CE→NDCG=0.59, BPR→NDCG=0.48, 禁止使用BPR!
2. [验证✓] 用户特征融合(u_cat_01~08)有效: 冷启动用户+0.01 (0.480→0.490)
3. [验证✓] 序列增强(随机截断至5/10)有效: 36K→89K训练样本
4. [验证✗] 流行度混合有害: 尺度不匹配, NDCG从0.48暴跌至0.26
5. [验证✗] 物品协同过滤(CF)无效: 共现相似度噪声大, 0.480→0.477
6. [验证✗] 滑动窗口增强有害: 产生过多噪声短序列, 0.480→0.455
7. [验证✓] GRU4Rec在测试集上优于SASRec: SASRec验证好但测试差(过拟合)
8. [验证✓] 物品特征(item.csv)尚未使用: i_cat_01~03, i_bucket_01可用
9. [关键洞察] 验证-测试差距大(0.59→0.49): 训练序列长(46)而测试短(6.25)
10. [验证✓] 增大模型容量未必有效: 边际收益递减
"""


class SOP:
    """标准操作流程: 指导Agent的每一步决策"""

    CLASSIFICATION_SOP = """
## 分类任务SOP (必须遵循)

### 实验配置规范
1. 模型: 必须使用GCN (不可用SAGE/GAT/APPNP, 除非有充分理由)
2. 归一化: feat_norm="l2" + normalization="sym"
3. DropEdge: drop_edge_rate=0.2 (始终启用, 不可关闭!)
4. 集成: n_ensemble=10-20
5. 标签传播: label_prop_alpha=0.3-0.4
6. 超参基线: hidden_dim=256, num_layers=2, dropout=0.5, lr=0.01, weight_decay=5e-4

### 诊断决策规则 (IF-THEN)
- IF 孤立节点比例>30% THEN 提高标签传播权重(0.3→0.4)
- IF 验证准确率<0.65 THEN 检查DropEdge是否启用, 检查feat_norm是否为l2
- IF 首轮baseline完成且准确率>0.65 THEN 考虑伪标签(置信度>0.8)
- IF 伪标签后准确率下降 THEN 降低置信度阈值或回退
- IF 2轮提升<0.005 THEN 输出STOP_AND_ENSEMBLE

### 禁止事项
- 禁止使用KNN图增强(已验证崩溃)
- 禁止关闭DropEdge
- 禁止使用随机游走归一化(rw)
"""

    RECOMMENDATION_SOP = """
## 推荐任务SOP (必须遵循)

### 实验配置规范
1. 模型: GRU4Rec + 用户特征融合 (GRU4RecWithFeatures)
2. 损失: 必须使用CE (禁止BPR!)
3. 用户特征: 必须融合u_cat_01~08 (8个类别特征)
4. 序列增强: use_seq_aug=True (随机截断至5/10)
5. 超参基线: embed=64, hidden=128, layers=1, dropout=0.2, lr=0.001, max_seq=50

### 诊断决策规则 (IF-THEN)
- IF 冷启动NDCG(seq_len<5) < 暖用户NDCG - 0.05 THEN 加强用户特征融合
- IF 整体NDCG < 0.50 THEN 检查是否误用BPR, 检查是否启用了序列增强
- IF 首轮完成且NDCG>0.55 THEN 考虑添加物品特征(item.csv)
- IF 2轮提升<0.005 THEN 输出STOP_AND_ENSEMBLE

### 禁止事项
- 禁止使用BPR损失 (已验证远差于CE)
- 禁止使用流行度混合 (已验证崩溃)
- 禁止使用滑动窗口增强 (已验证有害)
- 禁止使用物品协同过滤 (已验证无效)
"""

    BUDGET_SOP = """
## 预算管理SOP
1. 每轮实验后报告: 已用时间 / 总预算(120分钟) / 剩余时间
2. 决策规则:
   - IF 剩余时间 < 15分钟 THEN 输出 STOP_AND_ENSEMBLE
   - IF 连续2轮提升 < 0.005 THEN 考虑 STOP_AND_ENSEMBLE
   - IF 首轮实验失败(指标低于基线) THEN 不计入有效轮次, 继续探索
3. 停止时: 选择历史最佳模型的预测作为最终提交
"""

    CROSS_TASK_SOP = """
## 跨任务经验迁移SOP
1. 分类任务完成后, 提炼3条可迁移经验:
   - 容量选择: "中等容量(hidden=256)比大容量更稳定"
   - 正则化: "dropout=0.5对稀疏数据有效"
   - 集成: "多模型集成始终有效"
2. 将经验注入推荐任务的Agent提示词
3. 推荐Agent在决策时必须引用分类经验
4. 记录经验迁移在轨迹日志中
"""

    @classmethod
    def get_all_sops(cls):
        """获取所有SOP"""
        return {
            "classification": cls.CLASSIFICATION_SOP,
            "recommendation": cls.RECOMMENDATION_SOP,
            "budget": cls.BUDGET_SOP,
            "cross_task": cls.CROSS_TASK_SOP,
        }

    @classmethod
    def build_system_prompt(cls, task_type):
        """构建Agent系统提示词: 角色 + 知识库 + SOP"""
        if task_type == "classification":
            knowledge = KnowledgeBase.CLASSIFICATION_LESSONS
            sop = cls.CLASSIFICATION_SOP
            task_desc = """## 任务信息
- 数据: 13,752节点, 767维特征, 10类, 图极度稀疏(平均度数2.04, 53%孤立节点)
- 训练11,001 / 测试2,751, 评测指标: Accuracy"""
        else:
            knowledge = KnowledgeBase.RECOMMENDATION_LESSONS
            sop = cls.RECOMMENDATION_SOP
            task_desc = """## 任务信息
- 数据: 50,000用户, 2,156物品, 训练用户序列均长46, 测试用户序列均长6.25
- 评测指标: NDCG@10, 核心挑战: 测试用户序列极短(冷启动)"""

        return f"""你是一位资深金融算法专家, 正在受控实验沙盒中自主开展多轮实验。

{task_desc}

{knowledge}

{sop}

{cls.BUDGET_SOP}

## 你的职责
1. 分析每轮实验的多维诊断反馈(整体指标+子群体指标)
2. 根据SOP和经验库决定下一轮实验方向
3. 在rationale中详细说明: 你看到了什么问题→你决定怎么改→为什么
4. 管理预算, 在合适时机输出STOP_AND_ENSEMBLE
5. 不要只调超参数! 要关注特征工程、模型架构、数据增强等高阶决策"""

    @classmethod
    def build_cross_task_prompt(cls, cross_task_experiences):
        """构建跨任务经验迁移提示词"""
        if not cross_task_experiences:
            return "暂无跨任务经验"
        lines = ["## 跨任务经验 (来自分类任务)"]
        for exp in cross_task_experiences:
            lines.append(f"- {exp}")
        lines.append("\n请在决策时参考以上经验, 并在rationale中说明如何应用。")
        return "\n".join(lines)
