# RAG评估完整指南

本指南将帮助你完成在GuardReasoner HSDPO模型上添加RAG（Retrieval-Augmented Generation），并评估加RAG和不加RAG的对比效果。

## 📋 目录

1. [什么是RAG？](#什么是rag)
2. [准备工作](#准备工作)
3. [完整步骤](#完整步骤)
4. [结果解读](#结果解读)
5. [常见问题](#常见问题)

---

## 🎯 什么是RAG？

**RAG (Retrieval-Augmented Generation)** 是一种增强大语言模型的方法：

1. **检索（Retrieval）**：从知识库中检索与当前输入最相关的示例
2. **增强（Augmentation）**：将检索到的示例添加到输入prompt中
3. **生成（Generation）**：模型基于增强后的prompt生成答案

在GuardReasoner中，RAG的作用是：
- 从训练数据中检索相似的推理示例
- 帮助模型更好地理解如何进行分类和推理
- 提高模型在WildGuardTest上的表现

---

## 🔧 准备工作

### 1. 检查依赖

确保已安装以下Python包：

```bash
pip install vllm sentence-transformers numpy pandas scikit-learn
```

### 2. 检查模型路径

确保你的HSDPO模型路径正确。默认使用HuggingFace模型：
- `6Amber6/GuardReasoner-1B-HS-DPO`

如果需要使用本地模型，请修改 `run_rag_evaluation.py` 中的 `HSDPO_MODEL_PATH`。

### 3. 检查数据文件

确保测试数据存在：
```bash
ls ./data/benchmark/0_4_wild_guard_test.json
```

---

## 🚀 完整步骤

### 方法一：一键运行（推荐）

最简单的方式是运行一键脚本：

```bash
python run_rag_evaluation.py
```

这个脚本会自动完成所有步骤：
1. ✅ 检查依赖
2. ✅ 构建RAG知识库
3. ✅ 生成不带RAG的预测
4. ✅ 生成带RAG的预测
5. ✅ 评估并对比结果

### 方法二：分步运行

如果你想更精细地控制每个步骤，可以手动运行：

#### 步骤1: 构建RAG知识库

```bash
python build_rag_knowledge_base.py
```

**说明**：
- 如果训练数据存在（`./data/WildGuardTrainR.json`等），会使用训练数据
- 如果训练数据不存在，会使用benchmark数据作为示例
- 知识库会保存在 `./rag_knowledge_base/knowledge_base.pkl`

**预期输出**：
```
Loading embedding model...
Processing ./data/benchmark/0_4_wild_guard_test.json...
Total knowledge items: 1725
Generating embeddings...
Knowledge base saved to ./rag_knowledge_base
```

#### 步骤2: 生成不带RAG的预测

```bash
python generate_wildguard.py \
    --model_path "6Amber6/GuardReasoner-1B-HS-DPO" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_NoRAG"
```

**说明**：
- 使用HSDPO模型生成预测，不使用RAG
- 结果保存在 `./data/test/1B/WildGuardTest_HSDPO_NoRAG/generated_predictions.jsonl`

#### 步骤3: 生成带RAG的预测

```bash
python generate_with_rag.py \
    --model_path "6Amber6/GuardReasoner-1B-HS-DPO" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_WithRAG" \
    --use_rag \
    --top_k 3
```

**说明**：
- 使用HSDPO模型生成预测，**使用RAG增强**
- `--top_k 3` 表示检索最相关的3个示例
- 结果保存在 `./data/test/1B/WildGuardTest_HSDPO_WithRAG/generated_predictions.jsonl`

#### 步骤4: 评估并对比

```bash
python evaluate_wildguard.py --folders \
    "./data/test/1B/WildGuardTest_HSDPO_NoRAG/" \
    "./data/test/1B/WildGuardTest_HSDPO_WithRAG/"
```

**说明**：
- 评估两个文件夹中的预测结果
- 会显示每个任务的F1、Precision、Recall、Accuracy
- 会显示Valid和Skipped样本数

---

## 📊 结果解读

### 评估指标说明

- **F1 Score**: 精确率和召回率的调和平均数，综合评估指标
- **Precision**: 精确率，预测为正例中真正为正例的比例
- **Recall**: 召回率，真正为正例中被正确预测的比例
- **Accuracy**: 准确率，所有预测中正确的比例
- **Valid**: 成功提取标签的样本数
- **Skipped**: 无法提取标签的样本数（越少越好）

### 三个任务

1. **Prompt Harmfulness Detection**: 检测用户请求是否有害
2. **Response Harmfulness Detection**: 检测AI回复是否有害
3. **Refusal Detection**: 检测AI是否拒绝回答

### 对比结果示例

```
【对比结果：RAG vs 无RAG】

📊 PROMPT 任务:
指标             无RAG           有RAG           提升
F1               72.50           75.20           +2.70
Precision        70.30           73.10           +2.80
Recall           74.90           77.50           +2.60
Accuracy         85.20           87.10           +1.90
Valid            1720            1722            +2
Skipped          5               3               -2
```

**解读**：
- 如果"提升"列是正数，说明RAG有帮助
- 如果"提升"列是负数，说明RAG可能没有帮助或需要调整
- Valid越多、Skipped越少越好

---

## ❓ 常见问题

### Q1: RAG知识库构建很慢怎么办？

**A**: 这是正常的，因为需要：
1. 加载embedding模型（首次会下载）
2. 为所有知识项生成embedding向量

**建议**：
- 首次运行会下载模型，需要网络连接
- 构建完成后，知识库会保存，下次不需要重新构建
- 如果训练数据很大，可以考虑只使用部分数据

### Q2: 生成预测时内存不足？

**A**: 代码已经默认使用较低的内存占用设置：
- `gpu_memory_utilization=0.7` (默认0.95，已降低)
- `max_num_seqs=128` (默认256，已降低)

**注意**：降低这些参数可能会稍微降低生成速度，但可以避免内存不足的错误。

### Q3: RAG没有提升效果怎么办？

**A**: 可能的原因和解决方案：

1. **知识库质量不高**
   - 检查训练数据是否包含高质量的推理示例
   - 尝试使用更多训练数据构建知识库

2. **top_k设置不当**
   - 尝试不同的top_k值（1, 3, 5, 10）
   - 在 `generate_with_rag.py` 中使用 `--top_k` 参数

3. **检索质量不高**
   - 检查检索到的示例是否真的相关
   - 可以添加调试代码查看检索到的示例

### Q4: 如何查看检索到的示例？

**A**: 可以在 `generate_with_rag.py` 中添加调试代码：

```python
# 在 generate_with_rag 函数中，检索后添加：
if i < 3:  # 只打印前3个样本
    print(f"\n样本 {i+1} 检索到的示例:")
    for j, item in enumerate(retrieved_items):
        print(f"  示例 {j+1}: {item['input'][:100]}...")
```

### Q5: 可以使用不同的embedding模型吗？

**A**: 可以，在 `build_rag_knowledge_base.py` 中修改：

```python
# 当前使用轻量级模型
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# 可以使用更强大的模型（但会更慢）
encoder = SentenceTransformer('all-mpnet-base-v2')
```

### Q6: 如何只评估特定任务？

**A**: 可以修改 `evaluate_wildguard.py`，或者直接查看输出结果中的特定任务部分。

---

## 📁 文件结构

完成所有步骤后，你的文件结构应该是：

```
GuardReasoner/
├── run_rag_evaluation.py          # 一键运行脚本（推荐）
├── build_rag_knowledge_base.py    # RAG知识库构建脚本
├── generate_wildguard.py          # 不带RAG的生成脚本
├── generate_with_rag.py           # 带RAG的生成脚本
├── evaluate_wildguard.py          # 评估脚本
├── rag_knowledge_base/            # RAG知识库目录
│   ├── knowledge_base.pkl        # 知识库文件
│   └── encoder/                   # embedding模型
└── data/
    └── test/
        └── 1B/
            ├── WildGuardTest_HSDPO_NoRAG/      # 不带RAG的预测
            │   └── generated_predictions.jsonl
            └── WildGuardTest_HSDPO_WithRAG/    # 带RAG的预测
                └── generated_predictions.jsonl
```

---

## 完成！

现在你已经完成了RAG的集成和评估。如果RAG提升了模型效果，恭喜你！如果效果不理想，可以尝试：

1. 调整top_k参数
2. 使用更多/更好的训练数据构建知识库
3. 尝试不同的embedding模型
4. 调整RAG上下文的格式



