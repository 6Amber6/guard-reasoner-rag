"""
诊断并修复预测结果格式问题
"""
import json
import os
import re

def diagnose_predictions(folder):
    """诊断预测文件的问题"""
    file_path = os.path.join(folder, "generated_predictions.jsonl")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"诊断: {folder}")
    print(f"{'='*80}\n")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总样本数: {len(lines)}\n")
    
    # 统计信息
    stats = {
        'total': len(lines),
        'has_answers': 0,
        'has_request': 0,
        'has_response': 0,
        'has_completion': 0,
        'too_short': 0,  # <50字符
        'too_long': 0,   # >3000字符
        'normal': 0      # 50-3000字符
    }
    
    # 检查前10个样本
    print("前10个样本的详细信息:")
    print("-" * 80)
    for i, line in enumerate(lines[:10], 1):
        data = json.loads(line)
        predict = data['predict']
        length = len(predict)
        
        stats['has_answers'] += 1 if "Answers:" in predict else 0
        stats['has_request'] += 1 if "Request:" in predict else 0
        stats['has_response'] += 1 if "Response:" in predict else 0
        stats['has_completion'] += 1 if "Completion:" in predict else 0
        
        if length < 50:
            stats['too_short'] += 1
            status = "❌ 太短"
        elif length > 3000:
            stats['too_long'] += 1
            status = "⚠️  太长（可能重复）"
        else:
            stats['normal'] += 1
            status = "✓ 正常"
        
        print(f"\n样本 {i}:")
        print(f"  长度: {length} 字符 {status}")
        print(f"  内容预览: {predict[:200]}...")
        print(f"  包含Answers:: {'✓' if 'Answers:' in predict else '✗'}")
        print(f"  包含Request:: {'✓' if 'Request:' in predict else '✗'}")
        print(f"  包含Response:: {'✓' if 'Response:' in predict else '✗'}")
        print(f"  包含Completion:: {'✓' if 'Completion:' in predict else '✗'}")
    
    # 统计信息
    print(f"\n{'='*80}")
    print("统计信息（前10个样本）:")
    print(f"  包含Answers:: {stats['has_answers']}/10")
    print(f"  包含Request:: {stats['has_request']}/10")
    print(f"  包含Response:: {stats['has_response']}/10")
    print(f"  包含Completion:: {stats['has_completion']}/10")
    print(f"  太短(<50): {stats['too_short']}/10")
    print(f"  太长(>3000): {stats['too_long']}/10")
    print(f"  正常: {stats['normal']}/10")
    
    # 给出建议
    print(f"\n{'='*80}")
    print("诊断建议:")
    print(f"{'='*80}")
    
    if stats['too_short'] > 0:
        print("❌ 发现问题：有样本输出太短（<50字符）")
        print("   可能原因：")
        print("   1. 模型输出格式错误")
        print("   2. 模型训练不充分")
        print("   3. 需要检查模型是否正确加载")
    
    if stats['too_long'] > 0:
        print("⚠️  发现问题：有样本输出太长（>3000字符，可能重复）")
        print("   已添加后处理逻辑，会自动提取第一个完整答案")
    
    if stats['has_answers'] < 10:
        print("❌ 发现问题：部分样本缺少'Answers:'部分")
        print("   这会导致标签提取失败")
    
    if stats['has_request'] < 10 or stats['has_response'] < 10 or stats['has_completion'] < 10:
        print("❌ 发现问题：部分样本缺少必要的标签")
        print("   这会导致评估时大量样本被跳过")


if __name__ == "__main__":
    import sys
    
    folders = [
        "./data/test/1B/WildGuardTest_RSFT_HF",
        "./data/test/1B/WildGuardTest_HSDPO_HF"
    ]
    
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    
    for folder in folders:
        diagnose_predictions(folder)

