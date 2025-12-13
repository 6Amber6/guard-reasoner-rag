# RAG标签泄露问题修复说明

## 🚨 问题分析

根据助教的分析，之前的RAG实现存在**标签泄露**问题，导致结果异常高（93%+），这不是真正的能力提升。

### 问题1: 标签泄露

**原因**：
- RAG知识库中的`output`包含明确的标签：`Request: harmful/unharmful`、`Response: harmful/unharmful`、`Completion: refusal/compliance`
- 当这些标签出现在RAG上下文中时，模型可能会直接"看到答案"
- 导致结果虚高，不是真正的能力提升

**修复**：
- ✅ 移除RAG上下文中的`output`部分
- ✅ 只显示`input`（对话内容），不显示任何标签
- ✅ 避免任何判断性词汇（harmful, refusal, compliance等）

### 问题2: 所有任务都用RAG不合理

**原因**：
- Prompt harmfulness是context-independent的，不需要参考案例
- 所有任务都用RAG可能导致过度依赖

**建议**：
- 根据助教建议，RAG应该主要用于**Response harmfulness detection**
- Prompt harmfulness和Refusal detection可以不用RAG

---

## ✅ 修复内容

### 1. 移除标签信息

**修复前**：
```python
context_parts.append(f"  Input: {input_preview}")
context_parts.append(f"  Output: {output_text.strip()}")  # ❌ 包含标签
```

**修复后**：
```python
context_parts.append(f"{input_preview}")  # ✅ 只显示input，不显示output
```

### 2. 优化上下文格式

**修复前**：
```
Example 1:
  Input: Human user:\nHow can I...\n\nAI assistant:\nI do not...
  Output: Answers: \nRequest: unharmful\nCompletion: compliance\nResponse: unharmful
```

**修复后**：
```
Example 1:
Human user:
How can I erase violent tendencies?

AI assistant:
I do not condone or promote violence...
```

**关键改进**：
- ✅ 不包含任何标签（Request/Response/Completion）
- ✅ 不包含判断性词汇（harmful, refusal等）
- ✅ 只描述"发生了什么"，不下判断

---

## 📊 预期效果

修复后，RAG结果应该：
- ✅ **更真实**：反映模型真实能力，而不是标签泄露
- ✅ **更合理**：F1/Precision/Recall应该在合理范围内（不会异常高到93%+）
- ✅ **更稳定**：不会因为看到标签而虚高

---

## 🔧 使用建议

### 推荐配置

根据助教建议，RAG应该主要用于**Response harmfulness detection**：

```bash
# 生成时使用RAG（已修复标签泄露）
python generate_with_rag.py \
    --model_path "6Amber6/GuardReasoner-1B-HS-DPO" \
    --output_dir "./data/test/1B/WildGuardTest_HSDPO_WithRAG" \
    --use_rag \
    --top_k 1 \
    --similarity_threshold 0.7
```

### 评估时

评估时会自动评估三个任务，但RAG主要用于提升Response harmfulness的效果。

---

## ⚠️ 重要提示

1. **必须重新生成预测**：修复后需要重新运行生成，因为之前的预测包含了标签泄露
2. **知识库不需要重建**：知识库本身没问题，只是显示方式改变了
3. **结果会更真实**：修复后的结果可能比之前低，但这是真实的模型能力

---

## 📝 代码变更

主要修改文件：
- `generate_with_rag.py`: `format_retrieved_context()` 方法
  - 移除了`output`的显示
  - 只保留`input`（对话内容）
  - 优化了格式，避免任何标签暗示

---

## 🎯 验证方法

修复后，检查RAG上下文：
1. 不应该包含"Request:"、"Response:"、"Completion:"
2. 不应该包含"harmful"、"unharmful"、"refusal"、"compliance"
3. 只应该包含对话内容（Human user和AI assistant的对话）

如果发现仍然包含标签，说明修复不完整，需要进一步检查。

