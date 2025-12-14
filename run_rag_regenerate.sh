#!/bin/bash

# Regenerate RAG predictions (delete old file and regenerate)

echo "=========================================="
echo "Regenerating RAG Predictions"
echo "=========================================="
echo ""

# Delete old prediction file
RAG_OUTPUT_DIR="./data/test/1B/WildGuardTest_HSDPO_WithRAG"
OLD_PRED_FILE="${RAG_OUTPUT_DIR}/generated_predictions.jsonl"

if [ -f "$OLD_PRED_FILE" ]; then
    echo "Deleting old prediction file: $OLD_PRED_FILE"
    rm "$OLD_PRED_FILE"
    echo "✓ Deleted"
else
    echo "Old file does not exist, skipping deletion"
fi

echo ""
echo "=========================================="
echo "Starting RAG Prediction Generation"
echo "=========================================="
echo ""

# Generate new predictions
python generate_with_rag.py \
    --model_path "6Amber6/GuardReasoner-1B-HS-DPO" \
    --output_dir "$RAG_OUTPUT_DIR" \
    --use_rag

echo ""
echo "=========================================="
echo "Generation Complete!"
echo "=========================================="
echo ""
echo "Next, you can run evaluation:"
echo "python evaluate_wildguard.py --folders \\"
echo "    ./data/test/1B/WildGuardTest_HSDPO_NoRAG/ \\"
echo "    ./data/test/1B/WildGuardTest_HSDPO_WithRAG/"

