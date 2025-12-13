#!/bin/bash

# 快速评估脚本
# 使用方法: bash quick_evaluate.sh

echo "=========================================="
echo "快速评估RAG结果"
echo "=========================================="
echo ""

# 检查文件是否存在
RAG_FILE="./data/test/1B/WildGuardTest_HSDPO_WithRAG/generated_predictions.jsonl"
NO_RAG_FILE="./data/test/1B/WildGuardTest_HSDPO_NoRAG/generated_predictions.jsonl"

if [ -f "$RAG_FILE" ]; then
    echo "✓ 找到带RAG的预测文件"
else
    echo "✗ 带RAG的预测文件不存在: $RAG_FILE"
    exit 1
fi

if [ -f "$NO_RAG_FILE" ]; then
    echo "✓ 找到不带RAG的预测文件"
    echo ""
    echo "将进行对比评估..."
    echo ""
    python evaluate_wildguard.py --folders \
        ./data/test/1B/WildGuardTest_HSDPO_NoRAG/ \
        ./data/test/1B/WildGuardTest_HSDPO_WithRAG/
else
    echo "✗ 不带RAG的预测文件不存在: $NO_RAG_FILE"
    echo ""
    echo "将只评估带RAG的结果..."
    echo ""
    python evaluate_wildguard.py --folders \
        ./data/test/1B/WildGuardTest_HSDPO_WithRAG/
fi

echo ""
echo "=========================================="
echo "评估完成！"
echo "=========================================="

