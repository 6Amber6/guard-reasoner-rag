"""
从HuggingFace加载模型并生成WildGuardTest预测结果
支持本地模型路径和HuggingFace模型路径
"""
import os
import json
from vllm import LLM, SamplingParams


def generate_from_model(model_path, output_dir, model_name=None):
    """
    从模型路径（本地或HuggingFace）生成预测结果
    
    Args:
        model_path: 模型路径，可以是：
                   - HuggingFace路径: "username/model-name"
                   - 本地路径: "saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b"
        output_dir: 输出目录
        model_name: 模型名称（用于显示）
    """
    if model_name is None:
        model_name = model_path.split('/')[-1]
    
    print("=" * 80)
    print(f"Generating predictions for: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # 加载模型
    print(f"\n[1/4] Loading model from {model_path}...")
    try:
        vllm_model = LLM(
            model=model_path, 
            gpu_memory_utilization=0.95, 
            max_num_seqs=256,
            trust_remote_code=True  # 如果需要自定义代码
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise
    
    sampling_params = SamplingParams(temperature=0., top_p=1., max_tokens=2048)
    
    # 加载测试数据
    data_path = "./data/benchmark/0_4_wild_guard_test.json"
    print(f"\n[2/4] Loading test data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} test samples")
    
    # 准备prompt列表
    print(f"\n[3/4] Preparing prompts...")
    prompt_list = []
    for sample in data:
        prompt = sample['instruction'] + "\n" + sample['input']
        prompt_list.append(prompt)
    print(f"✓ Prepared {len(prompt_list)} prompts")
    
    # 生成
    print(f"\n[4/4] Generating predictions (this may take a while)...")
    try:
        outputs = vllm_model.generate(prompt_list, sampling_params)
        print(f"✓ Generated {len(outputs)} predictions")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        del vllm_model
        raise
    
    # 保存结果
    print(f"\n[Saving] Writing predictions to file...")
    save_dict_list = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        save_dict = {
            "prompt": prompt, 
            "label": data[i]["output"], 
            "predict": generated_text
        }
        save_dict_list.append(save_dict)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, "generated_predictions.jsonl")
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in save_dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"✓ Predictions saved to: {file_path}")
    print(f"✓ Total samples: {len(save_dict_list)}")
    
    del vllm_model
    print("\n" + "=" * 80)
    print("Generation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate predictions from HuggingFace or local models"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Model path: HuggingFace (username/model-name) or local path"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        required=True,
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for display (optional)"
    )
    
    args = parser.parse_args()
    
    generate_from_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        model_name=args.model_name
    )

