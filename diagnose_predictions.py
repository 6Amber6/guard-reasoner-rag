"""
诊断预测结果文件，分析标签提取失败的原因
"""
import json
import pandas as pd
from evaluate_wildguard import extract_label

def diagnose_predictions(folder, task="response", num_samples=10):
    """
    诊断预测结果，找出标签提取失败的原因
    
    Args:
        folder: 预测结果文件夹路径
        task: 任务类型
        num_samples: 分析的样本数量
    """
    file_path = folder.rstrip("/") + "/generated_predictions.jsonl"
    
    try:
        preds = pd.read_json(file_path, lines=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    gt = pd.read_json("./data/benchmark/0_4_wild_guard_test.json")
    
    print("=" * 100)
    print(f"Diagnosing {folder} for task: {task}")
    print("=" * 100)
    print()
    
    # 统计信息
    total = min(len(preds), len(gt))
    extracted = 0
    failed = 0
    
    # 获取ground truth标签
    if task == "prompt":
        gt_key = "prompt_harm_label"
    elif task == "response":
        gt_key = "response_harm_label"
    else:
        gt_key = "response_refusal_label"
    
    failed_samples = []
    
    for i in range(min(total, num_samples)):
        pred_text = str(preds["predict"][i])
        gt_label = str(gt[gt_key][i]).lower().strip() if i < len(gt) else "unknown"
        
        extracted_label = extract_label(pred_text, task)
        
        if extracted_label:
            extracted += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"
            failed_samples.append({
                'index': i,
                'gt': gt_label,
                'pred_text': pred_text[-300:],  # Last 300 chars
                'full_text_length': len(pred_text)
            })
        
        if i < 5:  # Show first 5 samples
            print(f"Sample {i} [{status}]")
            print(f"  GT: {gt_label}")
            print(f"  Extracted: {extracted_label}")
            print(f"  Prediction (last 200 chars): {pred_text[-200:]}")
            print()
    
    print("=" * 100)
    print(f"Summary:")
    print(f"  Total samples checked: {min(total, num_samples)}")
    print(f"  Successfully extracted: {extracted}")
    print(f"  Failed to extract: {failed}")
    print("=" * 100)
    
    if failed_samples:
        print("\nFailed samples (showing first 3):")
        for sample in failed_samples[:3]:
            print(f"\nSample {sample['index']}:")
            print(f"  GT: {sample['gt']}")
            print(f"  Text length: {sample['full_text_length']}")
            print(f"  Last 300 chars:")
            print(f"  {sample['pred_text']}")
            print()
            
            # Try to find why it failed
            text_lower = sample['pred_text'].lower()
            if "response" in text_lower:
                print(f"  Contains 'response': Yes")
                if "harmful" in text_lower or "unharmful" in text_lower:
                    print(f"  Contains harmful/unharmful: Yes")
                    # Find the context
                    if "harmful" in text_lower:
                        idx = text_lower.find("harmful")
                        print(f"  Context around 'harmful': {sample['pred_text'][max(0,idx-30):idx+50]}")
                else:
                    print(f"  Contains harmful/unharmful: No")
            else:
                print(f"  Contains 'response': No")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True,
                       help="Folder containing generated_predictions.jsonl")
    parser.add_argument("--task", type=str, choices=["prompt", "response", "refusal"],
                       default="response",
                       help="Task to diagnose")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    diagnose_predictions(args.folder, args.task, args.num_samples)

