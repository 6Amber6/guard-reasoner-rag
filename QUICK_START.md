# GuardReasoner 快速开始指南

## 🚀 最常用的三个命令

### 1. 评估 WildGuardTest（NoRAG vs WithRAG）
```bash
python3 evaluate_wildguard.py --folders \
    ./data/test/1B/WildGuardTest_HSDPO_NoRAG/ \
    ./data/test/1B/WildGuardTest_HSDPO_WithRAG/
```

### 2. 生成 NoRAG 预测
```bash
python3 generate_wildguard.py \
    --model_path "6Amber6/GuardReasoner-1B-HS-DPO" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_NoRAG"
```

### 3. 生成 WithRAG 预测
```bash
python3 generate_with_rag.py \
    --model_path "6Amber6/GuardReasoner-1B-HS-DPO" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_WithRAG" \
    --use_rag
```

## 📋 完整工作流程

### 第一次使用（需要准备数据）

```bash
# 1. 下载训练数据（用于构建RAG知识库）
python3 download_train_data.py

# 2. 构建RAG知识库
python3 build_rag_knowledge_base.py

# 3. 生成NoRAG预测
python3 generate_wildguard.py \
    --model_path "6Amber6/GuardReasoner-1B-HS-DPO" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_NoRAG"

# 4. 生成WithRAG预测
python3 generate_with_rag.py \
    --model_path "6Amber6/GuardReasoner-1B-HS-DPO" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_WithRAG" \
    --use_rag

# 5. 评估并对比
python3 evaluate_wildguard.py --folders \
    ./data/test/1B/WildGuardTest_HSDPO_NoRAG/ \
    ./data/test/1B/WildGuardTest_HSDPO_WithRAG/
```

### 后续使用（数据已准备好）

```bash
# 直接运行评估即可
python3 evaluate_wildguard.py --folders \
    ./data/test/1B/WildGuardTest_HSDPO_NoRAG/ \
    ./data/test/1B/WildGuardTest_HSDPO_WithRAG/
```

## 📖 详细说明

查看 `CODE_EXPLANATION.md` 了解每个文件的详细说明。

