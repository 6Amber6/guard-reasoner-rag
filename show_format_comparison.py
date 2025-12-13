"""
清晰展示原作者的格式和当前代码的格式对比
"""
import json
import os

print("=" * 100)
print("格式对比：原作者的格式 vs 当前代码应该生成的格式")
print("=" * 100)

# 原作者的格式
original_file = "./data/test/1B/WildGuardTest/generated_predictions.jsonl"
if os.path.exists(original_file):
    print("\n【原作者的格式 - 样本1】")
    print("-" * 100)
    with open(original_file, 'r') as f:
        data = json.loads(f.readline())
        predict = data['predict']
        
        print("\n1. Prompt部分（前200字符）:")
        print(data['prompt'][:200] + "...")
        
        print("\n2. Predict部分 - 完整内容:")
        print(predict)
        
        print("\n3. Predict部分 - 最后300字符（评估脚本使用的部分）:")
        print(predict[-300:])
        
        print("\n4. 关键格式检查:")
        print(f"   - 包含 'Answers:': {'Answers:' in predict}")
        print(f"   - 包含 'Request:': {'Request:' in predict}")
        print(f"   - 包含 'Response:': {'Response:' in predict}")
        print(f"   - 包含 'Completion:': {'Completion:' in predict}")
        
        # 提取Answers部分
        if "Answers:" in predict:
            answers_idx = predict.find("Answers:")
            answers_section = predict[answers_idx:answers_idx+200]
            print("\n5. Answers部分（前200字符）:")
            print(answers_section)
        
        # 测试标签提取
        print("\n6. 标签提取测试（使用原作者的evaluate.py逻辑）:")
        import re
        pred_example = predict[-300:]
        try:
            request_match = re.search(r'Request:\s*(harmful|unharmful)', pred_example, re.IGNORECASE)
            response_match = re.search(r'Response:\s*(harmful|unharmful)', pred_example, re.IGNORECASE)
            completion_match = re.search(r'Completion:\s*(compliance|refusal)', pred_example, re.IGNORECASE)
            print(f"   - Request提取: {request_match.group(1) if request_match else 'FAILED'}")
            print(f"   - Response提取: {response_match.group(1) if response_match else 'FAILED'}")
            print(f"   - Completion提取: {completion_match.group(1) if completion_match else 'FAILED'}")
        except Exception as e:
            print(f"   - 提取失败: {e}")

print("\n" + "=" * 100)
print("【当前代码应该生成的格式】")
print("-" * 100)
print("""
根据 run_all.py 的代码，当前应该：

1. Prompt格式: sample['instruction'] + "\\n" + sample['input']
   ✅ 这与原作者的 generate.py 完全一致

2. 模型输出: 应该包含推理过程和 Answers 部分
   - 如果模型训练正确，应该输出类似原作者的格式
   - 包含 Task 1, Task 2, Task 3 的推理
   - 最后有 "Answers:" 部分

3. 标签提取: 使用最后300字符
   ✅ 已在 run_all.py 中修复

4. 正则表达式: r'Request:\\s*(harmful|unharmful)'
   ✅ 已在 run_all.py 中修复

如果当前生成的格式与原作者不同，可能的原因：
1. 模型输出格式不同（需要检查模型训练）
2. 模型没有正确学习到输出格式
3. 需要检查生成的predict字段内容
""")

print("\n" + "=" * 100)
print("运行以下命令查看当前生成的格式:")
print("=" * 100)
print("python compare_formats.py")
print("\n或者:")
print("head -1 ./data/test/1B/WildGuardTest_RSFT_HF/generated_predictions.jsonl | python -m json.tool")

