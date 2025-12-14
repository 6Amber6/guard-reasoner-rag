#!/bin/bash

# Quick start script for RAG evaluation
# Usage: bash quick_start_rag.sh

echo "=========================================="
echo "Quick Start: RAG Evaluation"
echo "=========================================="
echo ""
echo "This script will automatically complete the following steps:"
echo "1. Check dependencies"
echo "2. Build RAG knowledge base"
echo "3. Generate predictions without RAG (HSDPO model)"
echo "4. Generate predictions with RAG (HSDPO model)"
echo "5. Evaluate and compare results"
echo ""
echo "Estimated time: 10-30 minutes (depending on GPU and network speed)"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

python run_rag_evaluation.py

