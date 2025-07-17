# AutoGen vs LangGraph: Research Synthesis Comparison

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.ai)

A comprehensive comparison of AutoGen and LangGraph frameworks through a practical research synthesis implementation, featuring performance analytics and detailed analysis.

## 🎯 Key Findings

| Metric | AutoGen | LangGraph | Winner |
|--------|---------|-----------|---------|
| **Processing Time** | 119.27s | 167.12s | **AutoGen** |
| **API Calls** | 3 | 10 | **AutoGen** |
| **Workflow Steps** | 3 rounds | 5 nodes | - |
| **Approach** | Conversational | State Machine | - |

**AutoGen was 47.85 seconds faster** for this research synthesis task.

## 📊 Performance Analysis

### AutoGen Strengths
- ✅ **40% faster processing**
- ✅ **3x fewer API calls** 
- ✅ **Rapid prototyping** via conversational flow
- ✅ **Built-in document processing**

### LangGraph Strengths
- ✅ **Explicit state control**
- ✅ **Better debugging** via graph visualization
- ✅ **Production-ready** deterministic execution
- ✅ **Comprehensive state management**

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama with `granite3.3:8b` model
- Git

### Installation
```bash
git clone https://github.com/ThirtyNimrod/autogen-vs-langgraph-comparison.git
cd autogen-vs-langgraph-comparison

# Install dependencies
pip install -r src/autogen_implementation/requirements.txt
pip install -r src/langgraph_implementation/requirements.txt

# Ensure Ollama is running
ollama serve
ollama pull granite3.3:8b
```

### Run Comparison
```bash
# AutoGen implementation
python src/autogen_implementation/autogen_research_synthesis.py

# LangGraph implementation  
python src/langgraph_implementation/langgraph_research_synthesis.py

# Generate comparison analysis
python src/comparison_analysis/framework_comparison.py
```

## 🔬 Test Case

Both frameworks analyzed three academic papers on multi-agent AI systems using identical agents:
- **Nimrod**: Practical analyst (focuses on data and applications)
- **Kalkin**: Theoretical synthesizer (explores patterns and connections)  
- **Ezzgy**: Methodical coordinator (ensures comprehensive analysis)

**Input Documents:**
1. "The Rise of Graph-Based Orchestration for Enterprise Agents"
2. "Emergent Collaboration: A Study of Conversation-Driven Agentic Systems"
3. "Scalability and State in Multi-Agent AI: A Comparative Analysis"

## 📈 Results Summary

### AutoGen Results
- **Processing Time**: 119.27 seconds
- **Conversation Flow**: Natural agent interactions with emergent insights
- **API Efficiency**: 3 calls with streamlined token usage
- **Output Quality**: Comprehensive synthesis with collaboration patterns

### LangGraph Results  
- **Processing Time**: 167.12 seconds
- **Workflow Control**: Structured 5-node state machine execution
- **State Management**: Complete audit trail with explicit transitions
- **Output Quality**: Systematic analysis with detailed state tracking

## 🎯 When to Choose Each

### Choose AutoGen for:
- 🚀 Rapid prototyping and iteration
- 🎨 Creative problem-solving
- 📚 Research and experimentation
- 🔄 Educational applications

### Choose LangGraph for:
- 🏢 Production enterprise systems
- 📊 Complex state management
- 🔒 Compliance and auditability
- 📈 Mission-critical deployments

## 📁 Repository Structure

```
autogen-vs-langgraph-comparison/
├── src/
│   ├── autogen_implementation/
│   ├── langgraph_implementation/
│   └── comparison_analysis/
├── test_documents/
├── results/
│   ├── analytics/
│   ├── visualizations/
│   └── reports/
├── docs/
└── notebooks/
```

## 📊 Detailed Analysis

View the complete analysis:
- **Performance Report**: `results/reports/comprehensive_comparison_report.json`
- **Article Summary**: `results/reports/article_ready_summary.md`
- **Raw Results**: `results/analytics/autogen_results.json` & `langgraph_results.json`
- **Visualizations**: `results/visualizations/`

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Related Articles

- [AutoGen Documentation](https://microsoft.github.io/autogen/stable//index.html)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)

---

**⭐ If this comparison helped you choose the right framework, please star the repository!**