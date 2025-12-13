#!/bin/bash

# 使用原作者的generate.py风格，但使用严格格式prompt
# 生成两个HuggingFace模型的预测结果

# ============================================
# 配置：HuggingFace模型路径
# ============================================

# 模型1: R-SFT模型
MODEL1_HF_PATH="6Amber6/guardreasoner-1b-rsft"
MODEL1_OUTPUT_DIR="./data/test/1B/WildGuardTest_RSFT_HF"

# 模型2: HS-DPO模型
MODEL2_HF_PATH="6Amber6/GuardReasoner-1B-HS-DPO"
MODEL2_OUTPUT_DIR="./data/test/1B/WildGuardTest_HSDPO_HF"

# ============================================
# 生成预测结果（使用严格格式）
# ============================================

echo "=========================================="
echo "Generating predictions with STRICT FORMAT"
echo "Based on original generate.py style"
echo "=========================================="
echo ""

# 生成模型1的预测
echo ">>> Generating predictions for Model 1 (R-SFT)..."
python generate_wildguard_strict.py \
    --model_path "$MODEL1_HF_PATH" \
    --output_dir "$MODEL1_OUTPUT_DIR"

echo ""
echo ">>> Model 1 completed!"
echo ""

# 生成模型2的预测
echo ">>> Generating predictions for Model 2 (HS-DPO)..."
python generate_wildguard_strict.py \
    --model_path "$MODEL2_HF_PATH" \
    --output_dir "$MODEL2_OUTPUT_DIR"

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

