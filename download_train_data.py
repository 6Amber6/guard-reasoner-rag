"""
Download training data for RAG knowledge base
"""
import os
from datasets import load_dataset

def download_train_data(output_dir="./data"):
    """
    Download training data from HuggingFace and save to data directory
    
    Args:
        output_dir: Directory to save the JSON files
    """
    print("=" * 80)
    print("Downloading training data from HuggingFace")
    print("=" * 80)
    print()
    
    # Check if huggingface-cli is logged in
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✓ Logged in to HuggingFace as: {user.get('name', 'Unknown')}")
    except Exception as e:
        print("⚠️  Not logged in to HuggingFace")
        print("   Please run one of the following:")
        print("   1. python -m huggingface_hub login")
        print("   2. huggingface-cli login (if installed)")
        print("   3. Set HF_TOKEN environment variable")
        print()
        print("   You can also continue without login if the dataset is public,")
        print("   but login is recommended for better access.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    print()
    print("Loading dataset from yueliu1999/GuardReasonerTrain...")
    print("This may take a few minutes...")
    print()
    
    try:
        # Load dataset
        ds = load_dataset("yueliu1999/GuardReasonerTrain")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download each subset
        datasets_to_download = {
            "WildGuardTrainR": "WildGuardTrainR.json",
            "AegisTrainR": "AegisTrainR.json",
            "BeaverTailsTrainR": "BeaverTailsTrainR.json",
            "ToxicChatTrainR": "ToxicChatTrainR.json"
        }
        
        for subset_name, filename in datasets_to_download.items():
            print(f"Downloading {subset_name}...")
            filepath = os.path.join(output_dir, filename)
            ds[subset_name].to_json(filepath)
            print(f"✓ Saved to: {filepath}")
            print()
        
        print("=" * 80)
        print("✓ All training data downloaded successfully!")
        print("=" * 80)
        print()
        print("Files saved to:")
        for filename in datasets_to_download.values():
            print(f"  - {os.path.join(output_dir, filename)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print()
        print("Possible solutions:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify dataset access: https://huggingface.co/datasets/yueliu1999/GuardReasonerTrain")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download training data for RAG knowledge base")
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Directory to save the JSON files (default: ./data)")
    
    args = parser.parse_args()
    
    success = download_train_data(args.output_dir)
    
    if success:
        print()
        print("Next steps:")
        print("1. Verify files exist in ./data/ directory")
        print("2. Run: python build_rag_knowledge_base.py")
        print("   This will build the RAG knowledge base using the training data")
    else:
        exit(1)

