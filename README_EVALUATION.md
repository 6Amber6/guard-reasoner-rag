# 使用自己训练的模型进行评估

## 重要说明

**`evaluate_wildguard.py` 只读取预测结果文件，不直接加载模型。**

评估流程：
1. 使用模型生成预测结果 → `generate_wildguard.py`
2. 评估预测结果 → `evaluate_wildguard.py`

## 使用自己训练的模型

### 方法1：使用HS-DPO训练后的模型（推荐）

```bash
# 步骤1: 用你的模型生成预测结果
python generate_wildguard.py \
    --model_path saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b \
    --output_dir ./data/test/1B/WildGuardTest_MyModel

# 步骤2: 评估你的模型
python evaluate_wildguard.py --folders ./data/test/1B/WildGuardTest_MyModel/
```

### 方法2：使用R-SFT训练后的模型

```bash
# 步骤1: 用R-SFT模型生成预测结果
python generate_wildguard.py \
    --model_path saves/Llama-3.2-1B/full/guardreasoner_rsft_1b \
    --output_dir ./data/test/1B/WildGuardTest_RSFT

# 步骤2: 评估R-SFT模型
python evaluate_wildguard.py --folders ./data/test/1B/WildGuardTest_RSFT/
```

### 方法3：对比多个模型

```bash
# 生成不同模型的预测结果
python generate_wildguard.py --model_path saves/Llama-3.2-1B/full/guardreasoner_rsft_1b --output_dir ./data/test/1B/WildGuardTest_RSFT
python generate_wildguard.py --model_path saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b --output_dir ./data/test/1B/WildGuardTest_HSDPO

# 同时评估多个模型
python evaluate_wildguard.py --folders \
    ./data/test/1B/WildGuardTest_RSFT/ \
    ./data/test/1B/WildGuardTest_HSDPO/
```

## 如何确认模型路径

### 检查你的模型保存位置

```bash
# 查找所有训练好的模型
find . -type d -name "*guardreasoner*" 2>/dev/null

# 或者检查LLaMA-Factory的保存目录
ls -la saves/Llama-3.2-1B/full/
```

### 模型命名规则

根据训练脚本，模型通常保存在：
- **R-SFT**: `saves/Llama-3.2-1B/full/guardreasoner_rsft_1b`
- **HS-DPO**: `saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b`

## 常见问题

### Q: 如何知道当前评估的是哪个模型？

A: 检查预测结果文件的生成时间：
```bash
ls -lh ./data/test/1B/WildGuardTest/generated_predictions.jsonl
```

### Q: 模型路径不存在怎么办？

A: 
1. 检查你的训练脚本中的 `save_path` 参数
2. 确认模型是否训练完成
3. 使用绝对路径指定模型位置

### Q: 想评估原作者发布的模型怎么办？

A: 如果原作者发布了HuggingFace模型，可以这样使用：
```bash
python generate_wildguard.py \
    --model_path yueliu1999/GuardReasoner-1B \
    --output_dir ./data/test/1B/WildGuardTest_Original
```

## 完整示例

```bash
# 1. 确认你的模型路径
MY_MODEL_PATH="saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b"

# 2. 生成预测结果
python generate_wildguard.py \
    --model_path "$MY_MODEL_PATH" \
    --output_dir ./data/test/1B/WildGuardTest_MyModel

# 3. 评估结果
python evaluate_wildguard.py --folders ./data/test/1B/WildGuardTest_MyModel/
```

