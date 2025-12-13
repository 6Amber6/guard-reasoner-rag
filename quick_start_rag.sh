#!/bin/bash

# RAG评估快速开始脚本
# 使用方法: bash quick_start_rag.sh

echo "=========================================="
echo "RAG评估快速开始"
echo "=========================================="
echo ""
echo "这个脚本将自动完成以下步骤："
echo "1. 检查依赖"
echo "2. 构建RAG知识库"
echo "3. 生成不带RAG的预测（HSDPO模型）"
echo "4. 生成带RAG的预测（HSDPO模型）"
echo "5. 评估并对比结果"
echo ""
echo "预计时间：10-30分钟（取决于GPU和网络速度）"
echo ""
read -p "按Enter键开始，或Ctrl+C取消..."

python run_rag_evaluation.py

