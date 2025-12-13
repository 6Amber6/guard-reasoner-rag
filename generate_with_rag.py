"""
使用RAG增强的生成脚本，仅针对WildGuardTest
"""
import os
import json
import pickle
import numpy as np
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    """RAG检索器"""
    
    def __init__(self, knowledge_base_path="./rag_knowledge_base/knowledge_base.pkl", 
                 encoder_path="./rag_knowledge_base/encoder",
                 top_k=3):
        """
        初始化RAG检索器
        
        Args:
            knowledge_base_path: 知识库路径
            encoder_path: encoder路径
            top_k: 检索top-k个最相关的知识项
        """
        print("Loading RAG knowledge base...")
        with open(knowledge_base_path, 'rb') as f:
            self.knowledge_base = pickle.load(f)
        
        self.encoder = SentenceTransformer(encoder_path)
        self.top_k = top_k
        print(f"RAG knowledge base loaded with {len(self.knowledge_base['items'])} items")
    
    def retrieve(self, query_text):
        """
        检索最相关的知识项
        
        Args:
            query_text: 查询文本
            
        Returns:
            检索到的知识项列表
        """
        # 对查询文本进行编码
        query_embedding = self.encoder.encode([query_text])[0]
        
        # 计算相似度
        embeddings = self.knowledge_base['embeddings']
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 获取top-k
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        retrieved_items = []
        for idx in top_indices:
            retrieved_items.append(self.knowledge_base['items'][idx])
        
        return retrieved_items
    
    def format_retrieved_context(self, retrieved_items):
        """
        格式化检索到的上下文
        
        Args:
            retrieved_items: 检索到的知识项列表
            
        Returns:
            格式化后的上下文字符串
        """
        context_parts = []
        context_parts.append("## Relevant Examples for Reference:\n")
        
        for i, item in enumerate(retrieved_items, 1):
            context_parts.append(f"### Example {i}:")
            context_parts.append(f"Input: {item['input']}")
            if item['reasoning']:
                context_parts.append(f"Reasoning: {item['reasoning'][:500]}...")  # 限制长度
            context_parts.append("")
        
        return "\n".join(context_parts)


def generate_with_rag(model_path, output_dir="./data/test/1B/WildGuardTest_HSDPO_WithRAG", 
                      use_rag=True, top_k=3):
    """
    使用RAG增强生成预测结果
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        use_rag: 是否使用RAG
        top_k: RAG检索的top-k数量
    """
    # 加载模型
    print(f"Loading model from {model_path}...")
    # 降低GPU内存利用率以避免内存不足错误
    vllm_model = LLM(model=model_path, gpu_memory_utilization=0.7, max_num_seqs=128)
    sampling_params = SamplingParams(temperature=0., top_p=1., max_tokens=2048)
    
    # 加载RAG检索器
    rag_retriever = None
    if use_rag:
        try:
            rag_retriever = RAGRetriever(top_k=top_k)
        except Exception as e:
            print(f"Warning: Failed to load RAG retriever: {e}")
            print("Continuing without RAG...")
            use_rag = False
    
    # 加载测试数据
    data_path = "./data/benchmark/0_4_wild_guard_test.json"
    print(f"Loading test data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备prompt列表
    prompt_list = []
    
    for i, sample in enumerate(data):
        base_prompt = sample['instruction'] + "\n" + sample['input']
        
        # 如果使用RAG，添加检索到的上下文
        if use_rag and rag_retriever:
            # 从input中提取查询文本（用户请求和AI回复）
            query_text = sample['input']
            retrieved_items = rag_retriever.retrieve(query_text)
            context = rag_retriever.format_retrieved_context(retrieved_items)
            
            # 将上下文添加到prompt中
            enhanced_prompt = sample['instruction'] + "\n\n" + context + "\n" + sample['input']
            prompt_list.append(enhanced_prompt)
        else:
            prompt_list.append(base_prompt)
    
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
                       default="./data/test/1B/WildGuardTest_HSDPO_WithRAG",
                       help="Output directory for predictions")
    parser.add_argument("--use_rag", action="store_true", default=True,
                       help="Whether to use RAG")
    parser.add_argument("--no_rag", action="store_false", dest="use_rag",
                       help="Disable RAG")
    parser.add_argument("--top_k", type=int, default=3,
                       help="Number of retrieved examples for RAG")
    
    args = parser.parse_args()
    
    # 如果使用RAG，先检查知识库是否存在
    if args.use_rag:
        kb_path = "./rag_knowledge_base/knowledge_base.pkl"
        if not os.path.exists(kb_path):
            print("RAG knowledge base not found. Building it first...")
            from build_rag_knowledge_base import build_knowledge_base
            
            train_data_paths = [
                "./data/WildGuardTrainR.json",
                "./data/AegisTrainR.json",
                "./data/BeaverTailsTrainR.json",
                "./data/ToxicChatTrainR.json"
            ]
            
            # 如果训练数据不存在，使用benchmark数据
            if not any(os.path.exists(p) for p in train_data_paths):
                print("Training data not found. Using benchmark data to build knowledge base...")
                train_data_paths = ["./data/benchmark/0_4_wild_guard_test.json"]
            
            build_knowledge_base(train_data_paths)
    
    generate_with_rag(
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_rag=args.use_rag,
        top_k=args.top_k
    )

