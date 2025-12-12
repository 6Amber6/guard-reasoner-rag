#!/bin/bash

# WildGuardTest评估脚本
# 自动执行：生成预测（带/不带RAG）和评估

MODEL_PATH="saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b"
OUTPUT_DIR_BASE="./data/test/1B"

echo "=========================================="
echo "WildGuardTest Evaluation Pipeline"
echo "=========================================="
echo ""

# 步骤1: 生成不带RAG的预测
echo "Step 1: Generating predictions WITHOUT RAG..."
python generate_wildguard.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR_BASE/WildGuardTest"

echo ""
echo "Step 1 completed!"
echo ""

# 步骤2: 构建RAG知识库（如果不存在）
if [ ! -f "./rag_knowledge_base/knowledge_base.pkl" ]; then
    echo "Step 2: Building RAG knowledge base..."
    python build_rag_knowledge_base.py
    echo "Step 2 completed!"
else
    echo "Step 2: RAG knowledge base already exists, skipping..."
fi
echo ""

# 步骤3: 生成带RAG的预测
echo "Step 3: Generating predictions WITH RAG..."
python generate_with_rag.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR_BASE/WildGuardTest_RAG" \
    --use_rag \
    --top_k 3

echo ""
echo "Step 3 completed!"
echo ""

# 步骤4: 评估结果
echo "Step 4: Evaluating results..."
python evaluate_wildguard.py

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="

