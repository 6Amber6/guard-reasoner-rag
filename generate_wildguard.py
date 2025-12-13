"""
生成脚本：仅针对WildGuardTest，不使用RAG（用于对比）
"""
import os
import json
from vllm import LLM, SamplingParams


def generate_wildguard(model_path, output_dir="./data/test/1B/WildGuardTest"):
    """
    生成WildGuardTest的预测结果（不使用RAG）
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
    """
    # 加载模型
    print(f"Loading model from {model_path}...")
    # 降低GPU内存利用率以避免内存不足错误
    vllm_model = LLM(model=model_path, gpu_memory_utilization=0.7, max_num_seqs=128)
    sampling_params = SamplingParams(temperature=0., top_p=1., max_tokens=2048)
    
    # 加载测试数据
    data_path = "./data/benchmark/0_4_wild_guard_test.json"
    print(f"Loading test data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备prompt列表
    prompt_list = []
    for sample in data:
        prompt = sample['instruction'] + "\n" + sample['input']
        prompt_list.append(prompt)
    
    # 生成
    print("Generating predictions...")
    outputs = vllm_model.generate(prompt_list, sampling_params)
    
    # 保存结果
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
    
    print(f"Predictions saved to {file_path}")
    
    del vllm_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                       default="saves/Llama-3.2-1B/full/guardreasoner_hsdpo_1b",
                       help="Path to the trained model")
    parser.add_argument("--output_dir", type=str,
                       default="./data/test/1B/WildGuardTest",
                       help="Output directory for predictions")
    
    args = parser.parse_args()
    
    generate_wildguard(
        model_path=args.model_path,
        output_dir=args.output_dir
    )

