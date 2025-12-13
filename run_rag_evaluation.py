"""
RAG评估脚本：对比HSDPO模型加RAG和不加RAG的效果
使用方法: python run_rag_evaluation.py
"""

import os
import json
import subprocess
import sys
from pathlib import Path

# ===================================================
# 配置
# ===================================================
HSDPO_MODEL_PATH = "6Amber6/GuardReasoner-1B-HS-DPO"  # HuggingFace模型路径
OUTPUT_DIR_BASE = "./data/test/1B"
OUTPUT_DIR_NO_RAG = f"{OUTPUT_DIR_BASE}/WildGuardTest_HSDPO_NoRAG"
OUTPUT_DIR_WITH_RAG = f"{OUTPUT_DIR_BASE}/WildGuardTest_HSDPO_WithRAG"
RAG_KB_PATH = "./rag_knowledge_base/knowledge_base.pkl"


def print_step(step_num, step_name):
    """打印步骤标题"""
    print("\n" + "=" * 80)
    print(f"步骤 {step_num}: {step_name}")
    print("=" * 80 + "\n")


def check_dependencies():
    """检查依赖"""
    print_step(1, "检查依赖")
    
    required_packages = ['vllm', 'sentence_transformers', 'numpy', 'pandas', 'sklearn']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} 未安装")
    
    if missing:
        print(f"\n❌ 缺少以下依赖: {', '.join(missing)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ 所有依赖已安装")
    return True


def build_rag_knowledge_base():
    """构建RAG知识库"""
    print_step(2, "构建RAG知识库")
    
    if os.path.exists(RAG_KB_PATH):
        print(f"✓ RAG知识库已存在: {RAG_KB_PATH}")
        print("  如需重新构建，请删除该文件后重新运行")
        return True
    
    print("正在构建RAG知识库...")
    print("这可能需要几分钟时间...")
    
    try:
        from build_rag_knowledge_base import build_knowledge_base
        
        # 检查训练数据是否存在
        train_data_paths = [
            "./data/WildGuardTrainR.json",
            "./data/AegisTrainR.json",
            "./data/BeaverTailsTrainR.json",
            "./data/ToxicChatTrainR.json"
        ]
        
        # 如果训练数据不存在，使用benchmark数据
        if not any(os.path.exists(p) for p in train_data_paths):
            print("⚠️  训练数据不存在，使用benchmark数据构建知识库...")
            train_data_paths = ["./data/benchmark/0_4_wild_guard_test.json"]
        
        build_knowledge_base(train_data_paths)
        
        if os.path.exists(RAG_KB_PATH):
            print(f"✓ RAG知识库构建成功: {RAG_KB_PATH}")
            return True
        else:
            print(f"❌ RAG知识库构建失败: {RAG_KB_PATH} 不存在")
            return False
            
    except Exception as e:
        print(f"❌ 构建RAG知识库时出错: {e}")
        return False


def generate_without_rag():
    """生成不带RAG的预测"""
    print_step(3, "生成不带RAG的预测（HSDPO模型）")
    
    output_file = os.path.join(OUTPUT_DIR_NO_RAG, "generated_predictions.jsonl")
    if os.path.exists(output_file):
        print(f"✓ 不带RAG的预测已存在: {output_file}")
        print("  如需重新生成，请删除该文件后重新运行")
        return True
    
    print(f"使用模型: {HSDPO_MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR_NO_RAG}")
    
    try:
        from generate_wildguard import generate_wildguard
        generate_wildguard(
            model_path=HSDPO_MODEL_PATH,
            output_dir=OUTPUT_DIR_NO_RAG
        )
        
        if os.path.exists(output_file):
            print(f"✓ 不带RAG的预测生成成功: {output_file}")
            return True
        else:
            print(f"❌ 预测生成失败: {output_file} 不存在")
            return False
            
    except Exception as e:
        print(f"❌ 生成预测时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_with_rag():
    """生成带RAG的预测"""
    print_step(4, "生成带RAG的预测（HSDPO模型）")
    
    output_file = os.path.join(OUTPUT_DIR_WITH_RAG, "generated_predictions.jsonl")
    if os.path.exists(output_file):
        print(f"✓ 带RAG的预测已存在: {output_file}")
        print("  如需重新生成，请删除该文件后重新运行")
        return True
    
    if not os.path.exists(RAG_KB_PATH):
        print(f"❌ RAG知识库不存在: {RAG_KB_PATH}")
        print("   请先运行步骤2构建知识库")
        return False
    
    print(f"使用模型: {HSDPO_MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR_WITH_RAG}")
    print(f"RAG知识库: {RAG_KB_PATH}")
    
    try:
        from generate_with_rag import generate_with_rag as gen_rag
        gen_rag(
            model_path=HSDPO_MODEL_PATH,
            output_dir=OUTPUT_DIR_WITH_RAG,
            use_rag=True,
            top_k=3
        )
        
        if os.path.exists(output_file):
            print(f"✓ 带RAG的预测生成成功: {output_file}")
            return True
        else:
            print(f"❌ 预测生成失败: {output_file} 不存在")
            return False
            
    except Exception as e:
        print(f"❌ 生成预测时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_no_rag():
    """评估不带RAG的结果"""
    print_step(5, "评估不带RAG的HSDPO模型")
    
    no_rag_file = os.path.join(OUTPUT_DIR_NO_RAG, "generated_predictions.jsonl")
    
    if not os.path.exists(no_rag_file):
        print(f"❌ 不带RAG的预测文件不存在: {no_rag_file}")
        return None
    
    print("正在评估不带RAG的模型...")
    
    try:
        from evaluate_wildguard import evaluate_folder
        
        print("\n" + "=" * 80)
        print("【HSDPO模型评估结果（不带RAG）】")
        print("=" * 80)
        
        results_no_rag = {}
        task_names = {
            "prompt": "Prompt Harmfulness Detection",
            "response": "Response Harmfulness Detection",
            "refusal": "Refusal Detection"
        }
        
        for task in ["prompt", "response", "refusal"]:
            print(f"\n### {task_names[task]} ###")
            print("-" * 80)
            result = evaluate_folder(OUTPUT_DIR_NO_RAG, task)
            if result:
                results_no_rag[task] = {
                    "F1": result["f1"],
                    "Precision": result["precision"],
                    "Recall": result["recall"],
                    "Accuracy": result["accuracy"],
                    "Valid": result["valid"],
                    "Skipped": result["skipped"]
                }
                print(f"F1: {result['f1']:.2f} | Precision: {result['precision']:.2f} | "
                      f"Recall: {result['recall']:.2f} | Accuracy: {result['accuracy']:.2f} | "
                      f"Valid: {result['valid']} | Skipped: {result['skipped']}")
            else:
                print("评估失败")
        
        print("\n" + "=" * 80)
        print("不带RAG的评估完成！")
        print("=" * 80)
        
        return results_no_rag
        
    except Exception as e:
        print(f"❌ 评估时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_with_rag():
    """评估带RAG的结果"""
    print_step(6, "评估带RAG的HSDPO模型")
    
    with_rag_file = os.path.join(OUTPUT_DIR_WITH_RAG, "generated_predictions.jsonl")
    
    if not os.path.exists(with_rag_file):
        print(f"❌ 带RAG的预测文件不存在: {with_rag_file}")
        return None
    
    print("正在评估带RAG的模型...")
    
    try:
        from evaluate_wildguard import evaluate_folder
        
        print("\n" + "=" * 80)
        print("【HSDPO模型评估结果（带RAG）】")
        print("=" * 80)
        
        results_with_rag = {}
        task_names = {
            "prompt": "Prompt Harmfulness Detection",
            "response": "Response Harmfulness Detection",
            "refusal": "Refusal Detection"
        }
        
        for task in ["prompt", "response", "refusal"]:
            print(f"\n### {task_names[task]} ###")
            print("-" * 80)
            result = evaluate_folder(OUTPUT_DIR_WITH_RAG, task)
            if result:
                results_with_rag[task] = {
                    "F1": result["f1"],
                    "Precision": result["precision"],
                    "Recall": result["recall"],
                    "Accuracy": result["accuracy"],
                    "Valid": result["valid"],
                    "Skipped": result["skipped"]
                }
                print(f"F1: {result['f1']:.2f} | Precision: {result['precision']:.2f} | "
                      f"Recall: {result['recall']:.2f} | Accuracy: {result['accuracy']:.2f} | "
                      f"Valid: {result['valid']} | Skipped: {result['skipped']}")
            else:
                print("评估失败")
        
        print("\n" + "=" * 80)
        print("带RAG的评估完成！")
        print("=" * 80)
        
        return results_with_rag
        
    except Exception as e:
        print(f"❌ 评估时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(results_no_rag, results_with_rag):
    """对比两个结果"""
    print_step(7, "对比结果：RAG vs 无RAG")
    
    if results_no_rag is None or results_with_rag is None:
        print("❌ 无法对比：缺少评估结果")
        return False
    
    print("\n" + "=" * 80)
    print("【对比结果：RAG vs 无RAG】")
    print("=" * 80)
    
    tasks = ["prompt", "response", "refusal"]
    task_names = {
        "prompt": "Prompt Harmfulness Detection",
        "response": "Response Harmfulness Detection",
        "refusal": "Refusal Detection"
    }
    metrics = ["F1", "Precision", "Recall", "Accuracy"]
    
    for task in tasks:
        print(f"\n📊 {task_names[task]}")
        print("-" * 80)
        print(f"{'指标':<15} {'无RAG':<15} {'有RAG':<15} {'提升':<15}")
        print("-" * 80)
        
        for metric in metrics:
            no_rag_val = results_no_rag.get(task, {}).get(metric, 0)
            with_rag_val = results_with_rag.get(task, {}).get(metric, 0)
            improvement = with_rag_val - no_rag_val
            improvement_str = f"+{improvement:.2f}" if improvement >= 0 else f"{improvement:.2f}"
            
            print(f"{metric:<15} {no_rag_val:<15.2f} {with_rag_val:<15.2f} {improvement_str:<15}")
        
        # 显示Valid和Skipped
        no_rag_valid = results_no_rag.get(task, {}).get("Valid", 0)
        with_rag_valid = results_with_rag.get(task, {}).get("Valid", 0)
        no_rag_skipped = results_no_rag.get(task, {}).get("Skipped", 0)
        with_rag_skipped = results_with_rag.get(task, {}).get("Skipped", 0)
        
        print(f"{'Valid':<15} {no_rag_valid:<15} {with_rag_valid:<15} {with_rag_valid - no_rag_valid:<15}")
        print(f"{'Skipped':<15} {no_rag_skipped:<15} {with_rag_skipped:<15} {with_rag_skipped - no_rag_skipped:<15}")
    
    print("\n" + "=" * 80)
    print("对比完成！")
    print("=" * 80)
    
    return True


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("RAG评估流程：对比HSDPO模型加RAG和不加RAG的效果")
    print("=" * 80)
    
    # 步骤1: 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装缺失的依赖")
        sys.exit(1)
    
    # 步骤2: 构建RAG知识库
    if not build_rag_knowledge_base():
        print("\n❌ RAG知识库构建失败")
        sys.exit(1)
    
    # 步骤3: 生成不带RAG的预测
    if not generate_without_rag():
        print("\n❌ 不带RAG的预测生成失败")
        sys.exit(1)
    
    # 步骤4: 生成带RAG的预测
    if not generate_with_rag():
        print("\n❌ 带RAG的预测生成失败")
        sys.exit(1)
    
    # 步骤5: 评估不带RAG的结果
    results_no_rag = evaluate_no_rag()
    if results_no_rag is None:
        print("\n❌ 不带RAG的评估失败")
        sys.exit(1)
    
    # 步骤6: 评估带RAG的结果
    results_with_rag = evaluate_with_rag()
    if results_with_rag is None:
        print("\n❌ 带RAG的评估失败")
        sys.exit(1)
    
    # 步骤7: 对比结果
    if not compare_results(results_no_rag, results_with_rag):
        print("\n❌ 对比失败")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ 所有步骤完成！")
    print("=" * 80)
    print("\n结果文件位置:")
    print(f"  不带RAG: {OUTPUT_DIR_NO_RAG}/generated_predictions.jsonl")
    print(f"  带RAG:   {OUTPUT_DIR_WITH_RAG}/generated_predictions.jsonl")
    print(f"  RAG知识库: {RAG_KB_PATH}")


if __name__ == "__main__":
    main()

