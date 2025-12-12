#!/bin/bash

# 生成两个HuggingFace模型的预测结果
# 使用方法: bash generate_both_models.sh

# ============================================
# 配置：请修改为你的HuggingFace模型路径
# ============================================

# 模型1: R-SFT模型
MODEL1_HF_PATH="6Amber6/guardreasoner-1b-rsft"
MODEL1_OUTPUT_DIR="./data/test/1B/WildGuardTest_RSFT_HF"

# 模型2: HS-DPO模型
MODEL2_HF_PATH="6Amber6/GuardReasoner-1B-HS-DPO"
MODEL2_OUTPUT_DIR="./data/test/1B/WildGuardTest_HSDPO_HF"

# ============================================
# 生成预测结果
# ============================================

echo "=========================================="
echo "Generating predictions for both models"
echo "=========================================="
echo ""

# 生成模型1的预测
echo ">>> Generating predictions for Model 1 (R-SFT)..."
python generate_from_hf.py \
    --model_path "$MODEL1_HF_PATH" \
    --output_dir "$MODEL1_OUTPUT_DIR" \
    --model_name "R-SFT Model"

echo ""
echo ">>> Model 1 completed!"
echo ""

# 生成模型2的预测
echo ">>> Generating predictions for Model 2 (HS-DPO)..."
python generate_from_hf.py \
    --model_path "$MODEL2_HF_PATH" \
    --output_dir "$MODEL2_OUTPUT_DIR" \
    --model_name "HS-DPO Model"

echo ""
echo ">>> Model 2 completed!"
echo ""

# ============================================
# 评估两个模型
# ============================================

echo "=========================================="
echo "Evaluating both models"
echo "=========================================="
echo ""

python evaluate_wildguard.py --folders \
    "$MODEL1_OUTPUT_DIR/" \
    "$MODEL2_OUTPUT_DIR/"

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="

