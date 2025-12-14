"""
RAG evaluation script: Compare HSDPO model with and without RAG
Usage: python run_rag_evaluation.py
"""

import os
import json
import subprocess
import sys
from pathlib import Path

# ===================================================
# Configuration
# ===================================================
HSDPO_MODEL_PATH = "6Amber6/GuardReasoner-1B-HS-DPO"  # HuggingFace model path
OUTPUT_DIR_BASE = "./data/test/1B"
OUTPUT_DIR_NO_RAG = f"{OUTPUT_DIR_BASE}/WildGuardTest_HSDPO_NoRAG"
OUTPUT_DIR_WITH_RAG = f"{OUTPUT_DIR_BASE}/WildGuardTest_HSDPO_WithRAG"
RAG_KB_PATH = "./rag_knowledge_base/knowledge_base.pkl"


def print_step(step_num, step_name):
    """Print step title"""
    print("\n" + "=" * 80)
    print(f"Step {step_num}: {step_name}")
    print("=" * 80 + "\n")


def check_dependencies():
    """Check dependencies"""
    print_step(1, "Check Dependencies")
    
    required_packages = ['vllm', 'sentence_transformers', 'numpy', 'pandas', 'sklearn']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} not installed")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Please install with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def build_rag_knowledge_base():
    """Build RAG knowledge base"""
    print_step(2, "Build RAG Knowledge Base")
    
    if os.path.exists(RAG_KB_PATH):
        print(f"✓ RAG knowledge base already exists: {RAG_KB_PATH}")
        print("  To rebuild, delete this file and run again")
        return True
    
    print("Building RAG knowledge base...")
    print("This may take a few minutes...")
    
    try:
        from build_rag_knowledge_base import build_knowledge_base
        
        train_data_paths = [
            "./data/WildGuardTrainR.json",
            "./data/AegisTrainR.json",
            "./data/BeaverTailsTrainR.json",
            "./data/ToxicChatTrainR.json"
        ]
        
        if not any(os.path.exists(p) for p in train_data_paths):
            print("⚠️  Training data not found. Using benchmark data to build knowledge base...")
            train_data_paths = ["./data/benchmark/0_4_wild_guard_test.json"]
        
        build_knowledge_base(train_data_paths)
        
        if os.path.exists(RAG_KB_PATH):
            print(f"✓ RAG knowledge base built successfully: {RAG_KB_PATH}")
            return True
        else:
            print(f"❌ RAG knowledge base build failed: {RAG_KB_PATH} does not exist")
            return False
            
    except Exception as e:
        print(f"❌ Error building RAG knowledge base: {e}")
        return False


def generate_without_rag():
    """Generate predictions without RAG"""
    print_step(3, "Generate Predictions Without RAG (HSDPO Model)")
    
    output_file = os.path.join(OUTPUT_DIR_NO_RAG, "generated_predictions.jsonl")
    if os.path.exists(output_file):
        print(f"✓ No-RAG predictions already exist: {output_file}")
        print("  To regenerate, delete this file and run again")
        return True
    
    print(f"Using model: {HSDPO_MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR_NO_RAG}")
    
    try:
        from generate_wildguard import generate_wildguard
        generate_wildguard(
            model_path=HSDPO_MODEL_PATH,
            output_dir=OUTPUT_DIR_NO_RAG
        )
        
        if os.path.exists(output_file):
            print(f"✓ No-RAG predictions generated successfully: {output_file}")
            return True
        else:
            print(f"❌ Prediction generation failed: {output_file} does not exist")
            return False
            
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_with_rag():
    """Generate predictions with RAG"""
    print_step(4, "Generate Predictions With RAG (HSDPO Model)")
    
    output_file = os.path.join(OUTPUT_DIR_WITH_RAG, "generated_predictions.jsonl")
    if os.path.exists(output_file):
        print(f"✓ RAG predictions already exist: {output_file}")
        print("  To regenerate, delete this file and run again")
        return True
    
    if not os.path.exists(RAG_KB_PATH):
        print(f"❌ RAG knowledge base does not exist: {RAG_KB_PATH}")
        print("   Please run step 2 to build knowledge base first")
        return False
    
    print(f"Using model: {HSDPO_MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR_WITH_RAG}")
    print(f"RAG knowledge base: {RAG_KB_PATH}")
    
    try:
        from generate_with_rag import generate_with_rag as gen_rag
        gen_rag(
            model_path=HSDPO_MODEL_PATH,
            output_dir=OUTPUT_DIR_WITH_RAG,
            use_rag=True,
            top_k=2,
            similarity_threshold=0.5
        )
        
        if os.path.exists(output_file):
            print(f"✓ RAG predictions generated successfully: {output_file}")
            return True
        else:
            print(f"❌ Prediction generation failed: {output_file} does not exist")
            return False
            
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_no_rag():
    """Evaluate results without RAG"""
    print_step(5, "Evaluate HSDPO Model Without RAG")
    
    no_rag_file = os.path.join(OUTPUT_DIR_NO_RAG, "generated_predictions.jsonl")
    
    if not os.path.exists(no_rag_file):
        print(f"❌ No-RAG prediction file does not exist: {no_rag_file}")
        return None
    
    print("Evaluating model without RAG...")
    
    try:
        from evaluate_wildguard import evaluate_folder
        
        print("\n" + "=" * 80)
        print("[HSDPO Model Evaluation Results (Without RAG)]")
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
                print("Evaluation failed")
        
        print("\n" + "=" * 80)
        print("Evaluation without RAG completed!")
        print("=" * 80)
        
        return results_no_rag
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_with_rag():
    """Evaluate results with RAG"""
    print_step(6, "Evaluate HSDPO Model With RAG")
    
    with_rag_file = os.path.join(OUTPUT_DIR_WITH_RAG, "generated_predictions.jsonl")
    
    if not os.path.exists(with_rag_file):
        print(f"❌ RAG prediction file does not exist: {with_rag_file}")
        return None
    
    print("Evaluating model with RAG...")
    
    try:
        from evaluate_wildguard import evaluate_folder
        
        print("\n" + "=" * 80)
        print("[HSDPO Model Evaluation Results (With RAG)]")
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
                print("Evaluation failed")
        
        print("\n" + "=" * 80)
        print("Evaluation with RAG completed!")
        print("=" * 80)
        
        return results_with_rag
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(results_no_rag, results_with_rag):
    """Compare two results"""
    print_step(7, "Compare Results: RAG vs No RAG")
    
    if results_no_rag is None or results_with_rag is None:
        print("❌ Cannot compare: missing evaluation results")
        return False
    
    print("\n" + "=" * 80)
    print("[Comparison Results: RAG vs No RAG]")
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
        print(f"{'Metric':<15} {'No RAG':<15} {'With RAG':<15} {'Improvement':<15}")
        print("-" * 80)
        
        for metric in metrics:
            no_rag_val = results_no_rag.get(task, {}).get(metric, 0)
            with_rag_val = results_with_rag.get(task, {}).get(metric, 0)
            improvement = with_rag_val - no_rag_val
            improvement_str = f"+{improvement:.2f}" if improvement >= 0 else f"{improvement:.2f}"
            
            print(f"{metric:<15} {no_rag_val:<15.2f} {with_rag_val:<15.2f} {improvement_str:<15}")
        
        no_rag_valid = results_no_rag.get(task, {}).get("Valid", 0)
        with_rag_valid = results_with_rag.get(task, {}).get("Valid", 0)
        no_rag_skipped = results_no_rag.get(task, {}).get("Skipped", 0)
        with_rag_skipped = results_with_rag.get(task, {}).get("Skipped", 0)
        
        print(f"{'Valid':<15} {no_rag_valid:<15} {with_rag_valid:<15} {with_rag_valid - no_rag_valid:<15}")
        print(f"{'Skipped':<15} {no_rag_skipped:<15} {with_rag_skipped:<15} {with_rag_skipped - no_rag_skipped:<15}")
    
    print("\n" + "=" * 80)
    print("Comparison completed!")
    print("=" * 80)
    
    return True


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("RAG Evaluation Pipeline: Compare HSDPO Model With and Without RAG")
    print("=" * 80)
    
    if not check_dependencies():
        print("\n❌ Dependency check failed, please install missing dependencies")
        sys.exit(1)
    
    if not build_rag_knowledge_base():
        print("\n❌ RAG knowledge base build failed")
        sys.exit(1)
    
    if not generate_without_rag():
        print("\n❌ No-RAG prediction generation failed")
        sys.exit(1)
    
    if not generate_with_rag():
        print("\n❌ RAG prediction generation failed")
        sys.exit(1)
    
    results_no_rag = evaluate_no_rag()
    if results_no_rag is None:
        print("\n❌ No-RAG evaluation failed")
        sys.exit(1)
    
    results_with_rag = evaluate_with_rag()
    if results_with_rag is None:
        print("\n❌ RAG evaluation failed")
        sys.exit(1)
    
    if not compare_results(results_no_rag, results_with_rag):
        print("\n❌ Comparison failed")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ All steps completed!")
    print("=" * 80)
    print("\nResult file locations:")
    print(f"  No RAG: {OUTPUT_DIR_NO_RAG}/generated_predictions.jsonl")
    print(f"  With RAG: {OUTPUT_DIR_WITH_RAG}/generated_predictions.jsonl")
    print(f"  RAG knowledge base: {RAG_KB_PATH}")


if __name__ == "__main__":
    main()

