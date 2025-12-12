# 使用HuggingFace模型生成预测结果

本指南说明如何使用你在HuggingFace上训练的两个模型生成预测结果。

## 方法1：使用便捷脚本（推荐）

### 步骤1：修改脚本中的模型路径

编辑 `generate_both_models.sh`，将模型路径替换为你的HuggingFace模型：

```bash
# 模型1: R-SFT模型
MODEL1_HF_PATH="your-username/guardreasoner-rsft-1b"  # 替换为你的R-SFT模型
MODEL1_OUTPUT_DIR="./data/test/1B/WildGuardTest_RSFT_HF"

# 模型2: HS-DPO模型  
MODEL2_HF_PATH="your-username/guardreasoner-hsdpo-1b"  # 替换为你的HS-DPO模型
MODEL2_OUTPUT_DIR="./data/test/1B/WildGuardTest_HSDPO_HF"
```

### 步骤2：运行脚本

```bash
bash generate_both_models.sh
```

这个脚本会：
1. 自动生成两个模型的预测结果
2. 自动评估两个模型并对比结果

## 方法2：手动运行（更灵活）

### 生成单个模型的预测

```bash
# 生成R-SFT模型的预测
python generate_from_hf.py \
    --model_path "your-username/guardreasoner-rsft-1b" \
    --output_dir "./data/test/1B/WildGuardTest_RSFT_HF" \
    --model_name "My R-SFT Model"

# 生成HS-DPO模型的预测
python generate_from_hf.py \
    --model_path "your-username/guardreasoner-hsdpo-1b" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_HF" \
    --model_name "My HS-DPO Model"
```

### 评估两个模型

```bash
python evaluate_wildguard.py --folders \
    ./data/test/1B/WildGuardTest_RSFT_HF/ \
    ./data/test/1B/WildGuardTest_HSDPO_HF/
```

## 方法3：使用原有的generate_wildguard.py

原有的 `generate_wildguard.py` 也支持HuggingFace路径：

```bash
# 生成R-SFT模型预测
python generate_wildguard.py \
    --model_path "your-username/guardreasoner-rsft-1b" \
    --output_dir "./data/test/1B/WildGuardTest_RSFT_HF"

# 生成HS-DPO模型预测
python generate_wildguard.py \
    --model_path "your-username/guardreasoner-hsdpo-1b" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_HF"
```

## HuggingFace模型路径格式

HuggingFace模型路径格式：
```
username/model-name
```

例如：
- `litong/guardreasoner-rsft-1b`
- `litong/guardreasoner-hsdpo-1b`
- `your-org/guardreasoner-1b`

## 完整示例

假设你的HuggingFace用户名是 `litong`，模型名称是 `guardreasoner-rsft-1b` 和 `guardreasoner-hsdpo-1b`：

```bash
# 1. 生成R-SFT模型预测
python generate_from_hf.py \
    --model_path "litong/guardreasoner-rsft-1b" \
    --output_dir "./data/test/1B/WildGuardTest_RSFT"

# 2. 生成HS-DPO模型预测
python generate_from_hf.py \
    --model_path "litong/guardreasoner-hsdpo-1b" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO"

# 3. 评估并对比
python evaluate_wildguard.py --folders \
    ./data/test/1B/WildGuardTest_RSFT/ \
    ./data/test/1B/WildGuardTest_HSDPO/
```

## 注意事项

1. **首次下载模型**：如果模型在HuggingFace上，vLLM会自动下载。确保：
   - 已登录HuggingFace：`huggingface-cli login`
   - 或者设置环境变量：`export HF_TOKEN=your_token`

2. **GPU内存**：确保有足够的GPU内存（建议至少8GB）

3. **网络连接**：首次下载模型需要稳定的网络连接

4. **模型格式**：确保你的HuggingFace模型包含：
   - `config.json`
   - `pytorch_model.bin` 或 `model.safetensors`
   - `tokenizer.json` 和 `tokenizer_config.json`

## 故障排除

### 问题1：模型下载失败

```bash
# 手动登录HuggingFace
huggingface-cli login

# 或者设置token
export HF_TOKEN=your_huggingface_token
```

### 问题2：模型路径错误

检查模型是否存在于HuggingFace：
```bash
# 在浏览器中访问
https://huggingface.co/your-username/your-model-name
```

### 问题3：权限问题

如果模型是私有的，确保：
1. 已登录HuggingFace
2. 有访问该模型的权限

### 问题4：内存不足

减少batch size：
```python
# 在generate_from_hf.py中修改
vllm_model = LLM(
    model=model_path, 
    gpu_memory_utilization=0.8,  # 降低到0.8
    max_num_seqs=128,  # 减少到128
)
```

## 输出文件结构

生成后的文件结构：
```
data/test/1B/
├── WildGuardTest_RSFT_HF/
│   └── generated_predictions.jsonl
└── WildGuardTest_HSDPO_HF/
    └── generated_predictions.jsonl
```

## 下一步

生成预测结果后，可以：
1. 评估模型性能：`python evaluate_wildguard.py --folders ...`
2. 对比不同模型的表现
3. 分析错误案例

