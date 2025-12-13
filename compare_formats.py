"""
对比原作者的格式和当前生成的格式
"""
import json
import os

print("=" * 100)
print("格式对比分析")
print("=" * 100)

# 1. 原作者的格式
original_file = "./data/test/1B/WildGuardTest/generated_predictions.jsonl"
if os.path.exists(original_file):
    print("\n【原作者生成的格式】")
    print("-" * 100)
    with open(original_file, 'r') as f:
        lines = f.readlines()[:3]
        for i, line in enumerate(lines, 1):
            data = json.loads(line)
            predict = data['predict']
            print(f"\n样本 {i}:")
            print(f"  总长度: {len(predict)} 字符")
            print(f"  最后300字符:")
            print(f"  {predict[-300:]}")
            print(f"  包含Answers:: {'Answers:' in predict}")
            print(f"  包含Request:: {'Request:' in predict}")
            print(f"  包含Response:: {'Response:' in predict}")
            print(f"  包含Completion:: {'Completion:' in predict}")
else:
    print(f"\n❌ 原作者的预测文件不存在: {original_file}")

# 2. 当前生成的格式 - RSFT
rsft_file = "./data/test/1B/WildGuardTest_RSFT_HF/generated_predictions.jsonl"
if os.path.exists(rsft_file):
    print("\n\n【当前生成的格式 - RSFT模型】")
    print("-" * 100)
    with open(rsft_file, 'r') as f:
        lines = f.readlines()[:3]
        for i, line in enumerate(lines, 1):
            data = json.loads(line)
            predict = data['predict']
            print(f"\n样本 {i}:")
            print(f"  总长度: {len(predict)} 字符")
            print(f"  最后300字符:")
            print(f"  {predict[-300:]}")
            print(f"  包含Answers:: {'Answers:' in predict}")
            print(f"  包含Request:: {'Request:' in predict}")
            print(f"  包含Response:: {'Response:' in predict}")
            print(f"  包含Completion:: {'Completion:' in predict}")
else:
    print(f"\n❌ RSFT预测文件不存在: {rsft_file}")
    print("   请先运行: python run_all.py")

# 3. 当前生成的格式 - HSDPO
hsdpo_file = "./data/test/1B/WildGuardTest_HSDPO_HF/generated_predictions.jsonl"
if os.path.exists(hsdpo_file):
    print("\n\n【当前生成的格式 - HSDPO模型】")
    print("-" * 100)
    with open(hsdpo_file, 'r') as f:
        lines = f.readlines()[:3]
        for i, line in enumerate(lines, 1):
            data = json.loads(line)
            predict = data['predict']
            print(f"\n样本 {i}:")
            print(f"  总长度: {len(predict)} 字符")
            print(f"  最后300字符:")
            print(f"  {predict[-300:]}")
            print(f"  包含Answers:: {'Answers:' in predict}")
            print(f"  包含Request:: {'Request:' in predict}")
            print(f"  包含Response:: {'Response:' in predict}")
            print(f"  包含Completion:: {'Completion:' in predict}")
else:
    print(f"\n❌ HSDPO预测文件不存在: {hsdpo_file}")
    print("   请先运行: python run_all.py")

print("\n" + "=" * 100)
print("对比总结")
print("=" * 100)
print("""
关键检查点：
1. 是否包含 "Answers:" 部分？
2. 是否包含 "Request:", "Response:", "Completion:" 标签？
3. 标签格式是否正确（冒号后跟空格）？
4. 最后300字符是否包含完整的答案部分？
""")

