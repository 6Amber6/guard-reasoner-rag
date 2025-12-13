# RAG优化总结

## 📊 原始问题分析

根据你的评估结果，RAG效果不如无RAG的主要原因：

### 1. Skip增加
- **Prompt**: 70 → 123 (+53)
- **Response**: 57 → 113 (+56)
- **原因**: RAG添加的上下文可能干扰了模型输出格式

### 2. Precision大幅下降
- **Prompt**: 71.70 → 55.30 (-16.40)
- **Refusal**: 69.33 → 57.05 (-12.28)
- **原因**: 检索到的案例可能质量不高，误导模型

### 3. Recall提升但F1下降
- **Response**: Recall +26.85, 但F1只+7.05
- **原因**: 找到了更多正例，但误判也增加了

---

## 🔧 优化措施

### 1. 添加相似度阈值过滤
**问题**: 检索到的案例可能相似度很低，质量不高

**解决方案**:
```python
similarity_threshold=0.5  # 只保留相似度>=0.5的案例
```

**效果**: 过滤掉低质量案例，只使用真正相似的参考案例

---

### 2. 减少Top-K数量
**问题**: Top-K=3可能太多，增加了干扰

**解决方案**:
```python
top_k=2  # 从3减少到2
```

**效果**: 减少上下文长度，降低对模型输出的干扰

---

### 3. 优化Prompt格式
**问题**: 原始格式可能不够清晰，模型不知道如何使用参考案例

**优化前**:
```
## Relevant Examples for Reference:

### Example 1:
Input: ...
Output: ...
```

**优化后**:
```
Here are some similar examples to help you understand the task:

Example 1:
  Input: ... (前200字符)
  Output: ...

Now analyze the following case:
```

**效果**: 
- 更清晰的指导语言
- 限制input长度，避免过长
- 明确指示"现在分析以下案例"

---

### 4. 智能回退机制
**问题**: 如果没有检索到相似案例，仍然使用RAG可能有害

**解决方案**:
```python
if retrieved_items:
    # 使用RAG
    enhanced_prompt = instruction + context + input
else:
    # 回退到无RAG
    base_prompt = instruction + input
```

**效果**: 只在有高质量参考案例时才使用RAG

---

### 5. 增强检索策略
**问题**: 只用input检索可能不够准确

**解决方案**:
```python
# 使用instruction的关键部分增强检索
instruction_key = "analyzing interactions between humans and AI"
enhanced_query = f"{instruction_key}\n{query_text}"
```

**效果**: 提高检索的准确性

---

## 📈 预期改进

### 预期效果：
1. **Skip减少**: 通过相似度阈值和格式优化，减少skip
2. **Precision提升**: 过滤低质量案例，提高精确度
3. **F1提升**: 在保持Recall的同时，提升Precision，从而提升F1

### 参数调整建议：

如果效果仍不理想，可以尝试：

1. **进一步减少Top-K**:
   ```bash
   python generate_with_rag.py --top_k 1
   ```

2. **提高相似度阈值**:
   ```bash
   python generate_with_rag.py --similarity_threshold 0.6
   ```

3. **完全禁用RAG**（如果效果仍然不好）:
   ```bash
   python generate_with_rag.py --no_rag
   ```

---

## 🚀 重新运行评估

运行优化后的RAG评估：

```bash
python run_rag_evaluation.py
```

或者手动调整参数：

```bash
python generate_with_rag.py \
    --model_path "6Amber6/GuardReasoner-1B-HS-DPO" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_WithRAG" \
    --use_rag \
    --top_k 2 \
    --similarity_threshold 0.5
```

---

## 📝 关键改进点总结

| 优化项 | 原始值 | 优化值 | 原因 |
|--------|--------|--------|------|
| Top-K | 3 | 2 | 减少干扰 |
| 相似度阈值 | 无 | 0.5 | 过滤低质量案例 |
| Prompt格式 | 简单 | 优化 | 更清晰的指导 |
| 检索策略 | 只用input | input+instruction关键部分 | 提高准确性 |
| 回退机制 | 无 | 有 | 无相似案例时不使用RAG |

---

## ⚠️ 注意事项

1. **知识库质量很重要**: 如果知识库本身质量不高，RAG效果会受限
2. **相似度阈值需要调优**: 0.5是默认值，可能需要根据实际情况调整
3. **Top-K需要平衡**: 太少可能信息不足，太多可能干扰模型

---

## 🔍 进一步优化方向

如果优化后效果仍不理想，可以考虑：

1. **使用更好的embedding模型**: 如 `all-mpnet-base-v2`（更准确但更慢）
2. **改进知识库构建**: 使用更高质量的训练数据
3. **任务特定的RAG**: 为每个任务（Prompt/Response/Refusal）构建独立的知识库
4. **混合检索**: 结合关键词检索和向量检索

