#!/bin/bash

# Copy Existing Files to Repository Structure
# This script moves your existing files to the proper GitHub repository structure

echo "📁 Copying existing files to repository structure..."

# Ensure we're in the repository directory
cd autogen-vs-langgraph-comparison

# Copy source files
echo "📄 Copying source files..."
cp ../autogen_research_synthesis.py src/autogen_implementation/
cp ../langgraph_research_synthesis.py src/langgraph_implementation/
cp ../framework_comparison.py src/comparison_analysis/

# Copy test documents
echo "📚 Copying test documents..."
cp ../test_documents/*.md test_documents/ 2>/dev/null || echo "⚠️  Test documents not found, creating placeholders..."

# Create test documents if they don't exist
if [ ! -f test_documents/test_document_1.md ]; then
    cat > test_documents/test_document_1.md << 'EOF'
# The Rise of Graph-Based Orchestration for Enterprise Agents

**Authors:** Dr. Alistair Finch, Dr. Evelyn Reed  
**Institution:** Institute for Enterprise AI (IEA), London  
**Date:** March 15, 2025

## Abstract

As multi-agent AI systems move from theoretical constructs to production environments, the need for robust, reliable, and auditable orchestration has become paramount. This paper argues that graph-based architectures, which model workflows as explicit state machines, represent the only viable path forward for mission-critical enterprise applications. We present a case study of FlowCorp, a global logistics firm, which replaced its legacy event-driven system with a graph-based agentic workflow for supply chain management. The results demonstrate a 95% reduction in reconciliation errors and a