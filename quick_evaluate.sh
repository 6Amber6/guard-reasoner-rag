#!/bin/bash

# Quick evaluation script
# Usage: bash quick_evaluate.sh

echo "=========================================="
echo "Quick RAG Evaluation"
echo "=========================================="
echo ""

# Check if files exist
RAG_FILE="./data/test/1B/WildGuardTest_HSDPO_WithRAG/generated_predictions.jsonl"
NO_RAG_FILE="./data/test/1B/WildGuardTest_HSDPO_NoRAG/generated_predictions.jsonl"

if [ -f "$RAG_FILE" ]; then
    echo "✓ Found RAG prediction file"
else
    echo "✗ RAG prediction file does not exist: $RAG_FILE"
    exit 1
fi

if [ -f "$NO_RAG_FILE" ]; then
    echo "✓ Found No-RAG prediction file"
    echo ""
    echo "Running comparison evaluation..."
    echo ""
    python evaluate_wildguard.py --folders \
        ./data/test/1B/WildGuardTest_HSDPO_NoRAG/ \
        ./data/test/1B/WildGuardTest_HSDPO_WithRAG/
else
    echo "✗ No-RAG prediction file does not exist: $NO_RAG_FILE"
    echo ""
    echo "Evaluating RAG results only..."
    echo ""
    python evaluate_wildguard.py --folders \
        ./data/test/1B/WildGuardTest_HSDPO_WithRAG/
fi

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="

