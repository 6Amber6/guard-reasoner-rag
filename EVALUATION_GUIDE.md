# WildGuardTest评估指南

本指南说明如何评估已训练的1B模型（R-SFT + HS-DPO）在WildGuardTest数据集上的表现，包括使用和不使用RAG的对比。

## 前置要求

1. 已安装必要的Python包：
```bash
pip install vllm sentence-transformers scikit-learn pandas numpy
```

2. 已训练好的模型位于：
   - `saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b`

## 步骤1：生成预测结果（不使用RAG）

首先，生成不使用RAG的预测结果作为baseline：

```bash
python generate_wildguard.py \
    --model_path saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b \
    --output_dir ./data/test/1B/WildGuardTest
```

## 步骤2：构建RAG知识库

在生成带RAG的预测之前，需要先构建RAG知识库：

```bash
python build_rag_knowledge_base.py
```

这个脚本会：
- 从训练数据中提取推理步骤作为知识库
- 如果训练数据不存在，会使用benchmark数据构建一个简单的知识库
- 生成向量embeddings用于检索
- 保存知识库到 `./rag_knowledge_base/`

## 步骤3：生成预测结果（使用RAG）

生成使用RAG增强的预测结果：

```bash
python generate_with_rag.py \
    --model_path saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b \
    --output_dir ./data/test/1B/WildGuardTest_RAG \
    --use_rag \
    --top_k 3
```

参数说明：
- `--model_path`: 模型路径
- `--output_dir`: 输出目录
- `--use_rag`: 启用RAG（默认启用）
- `--top_k`: 检索的top-k个最相关的示例（默认3）

如果知识库不存在，脚本会自动构建。

## 步骤4：评估结果

评估两个版本（带RAG和不带RAG）的性能：

```bash
# 评估所有任务
python evaluate_wildguard.py

# 评估特定任务
python evaluate_wildguard.py --task prompt    # Prompt harmfulness detection
python evaluate_wildguard.py --task response  # Response harmfulness detection
python evaluate_wildguard.py --task refusal    # Refusal detection
```

评估指标包括：
- F1 Score
- Precision
- Recall
- Accuracy

## 输出结果

评估结果会显示在终端，包括：
1. **Prompt Harmfulness Detection**: 检测用户请求是否有害
2. **Response Harmfulness Detection**: 检测AI回复是否有害
3. **Refusal Detection**: 检测AI回复是拒绝还是合规

每个任务都会显示带RAG和不带RAG的对比结果。

## 文件结构

生成的文件结构：
```
data/test/1B/
├── WildGuardTest/
│   └── generated_predictions.jsonl  # 不使用RAG的预测结果
└── WildGuardTest_RAG/
    └── generated_predictions.jsonl  # 使用RAG的预测结果

rag_knowledge_base/
├── knowledge_base.pkl  # 知识库（pickle格式）
└── encoder/            # Sentence transformer模型
```

## 注意事项

1. **GPU内存**: 确保有足够的GPU内存运行vLLM（建议至少8GB）
2. **知识库构建**: 如果训练数据不存在，脚本会使用benchmark数据构建知识库，但效果可能不如使用真实训练数据
3. **检索质量**: RAG的效果取决于知识库的质量和检索的top-k数量，可以根据需要调整`--top_k`参数

## 故障排除

1. **模型路径错误**: 检查模型路径是否正确
2. **知识库不存在**: 运行`build_rag_knowledge_base.py`手动构建
3. **内存不足**: 减少`gpu_memory_utilization`或`max_num_seqs`参数
4. **依赖包缺失**: 确保安装了所有必需的Python包

