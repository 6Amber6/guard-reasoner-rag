"""
使用数据集原始格式（instruction + input）生成WildGuardTest预测结果
这是论文作者使用的正确格式，与训练时完全一致

重要：必须使用 instruction + input 格式，而不是自定义的strict prompt
这样才能与训练时的格式匹配，模型才能正确输出
"""
import os
import json
from vllm import LLM, SamplingParams


def generate_wildguard_correct(model_path, output_dir="./data/test/1B/WildGuardTest"):
    """
    使用数据集原始格式（instruction + input）生成WildGuardTest预测结果
    这是论文作者使用的正确格式，与训练时一致
    
    Args:
        model_path: 模型路径（HuggingFace或本地）
        output_dir: 输出目录
    """
    # 加载模型（使用vllm，与原作者的代码一致）
    vllm_model = LLM(model=model_path, gpu_memory_utilization=0.95, max_num_seqs=256)
    sampling_params = SamplingParams(temperature=0., top_p=1., max_tokens=2048)  # 使用原始设置
    
    # 加载WildGuardTest数据
    data_name = "0_4_wild_guard_test"
    with open(f"./data/benchmark/{data_name}.json") as file:
        data = json.load(file)
    
    # 准备prompt列表（使用数据集原始格式：instruction + input）
    prompt_list = []
    for i, sample in enumerate(data):
        # 使用论文作者的正确格式：instruction + input
        # 这与训练时的格式完全一致
        prompt = sample['instruction'] + "\n" + sample['input']
        prompt_list.append(prompt)
    
    # 生成
    outputs = vllm_model.generate(prompt_list, sampling_params)
    
    # 保存结果
    save_dict_list = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        save_dict = {"prompt": prompt, "label": data[i]["output"], "predict": generated_text}
        save_dict_list.append(save_dict)
    
    # 创建输出目录
    file_path = os.path.join(output_dir, "generated_predictions.jsonl")
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 保存文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in save_dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Predictions saved to: {file_path}")
    print(f"Total samples: {len(save_dict_list)}")
    
    del vllm_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="6Amber6/guardreasoner-1b-rsft",
        help="Model path (HuggingFace or local)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./data/test/1B/WildGuardTest_RSFT_HF",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    generate_wildguard_correct(
        model_path=args.model_path,
        output_dir=args.output_dir
    )

