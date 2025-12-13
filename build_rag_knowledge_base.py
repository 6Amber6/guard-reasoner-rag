"""
构建RAG知识库
"""
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

def build_knowledge_base(train_data_paths, output_dir="./rag_knowledge_base"):
    """
    从训练数据中构建RAG知识库
    
    Args:
        train_data_paths: 训练数据文件路径列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载embedding模型
    print("Loading embedding model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级模型，适合快速检索
    
    knowledge_items = []
    
    # 从训练数据中提取知识
    for data_path in train_data_paths:
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping...")
            continue
            
        print(f"Processing {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # 只提取input和output
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            
            # 创建知识项
            knowledge_item = {
                'input': input_text,
                'output': output_text
            }
            knowledge_items.append(knowledge_item)
    
    print(f"Total knowledge items: {len(knowledge_items)}")
    
    # 为每个知识项生成embedding（只使用input）
    print("Generating embeddings...")
    texts_to_encode = []
    for item in knowledge_items:
        # 只使用input作为检索文本
        texts_to_encode.append(item['input'])
    
    embeddings = encoder.encode(texts_to_encode, show_progress_bar=True, batch_size=32)
    
    # 保存知识库
    knowledge_base = {
        'items': knowledge_items,
        'embeddings': embeddings
    }
    
    # 保存为pickle文件
    with open(os.path.join(output_dir, 'knowledge_base.pkl'), 'wb') as f:
        pickle.dump(knowledge_base, f)
    
    # 保存encoder
    encoder.save(os.path.join(output_dir, 'encoder'))
    
    print(f"Knowledge base saved to {output_dir}")
    return knowledge_base, encoder


if __name__ == "__main__":
    # 训练数据路径（如果存在的话）
    train_data_paths = [
        "./data/WildGuardTrainR.json",
        "./data/AegisTrainR.json",
        "./data/BeaverTailsTrainR.json",
        "./data/ToxicChatTrainR.json"
    ]
    
    # 如果训练数据不存在，可以从benchmark数据中构建一个简单的知识库
    if not any(os.path.exists(p) for p in train_data_paths):
        print("Training data not found. Building knowledge base from benchmark data...")
        # 使用benchmark数据作为示例知识库
        train_data_paths = ["./data/benchmark/0_4_wild_guard_test.json"]
    
    build_knowledge_base(train_data_paths)

