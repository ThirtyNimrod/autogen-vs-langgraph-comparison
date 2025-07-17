import asyncio
import json
import time
import os
import glob
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from pathlib import Path
import re
import uuid

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# LangChain imports for message handling
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableConfig

# Ollama integration
import requests

def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Add messages helper function"""
    return left + right

class ResearchState(TypedDict):
    """State schema for research synthesis workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Dict[str, Any]]
    processed_docs: List[Dict[str, Any]]
    nimrod_analysis: Dict[str, Any]
    kalkin_analysis: Dict[str, Any]
    ezzgy_analysis: Dict[str, Any]
    synthesis_result: Dict[str, Any]
    current_step: str
    processing_metrics: Dict[str, Any]
    error_log: List[str]
    thread_id: str  # Add thread_id for checkpointer

@dataclass
class LangGraphMetrics:
    """Metrics tracking for LangGraph workflows"""
    node_execution_times: Dict[str, List[float]]
    state_transitions: List[Dict[str, Any]]
    total_processing_time: float
    preprocessing_time: float
    api_calls: int
    error_count: int

class OllamaLLM(BaseChatModel):
    """Ollama LLM integration for LangGraph, designed for async workflows."""

    model_name: str = "granite3.3:8b"
    base_url: str = "http://localhost:11434"
    api_calls: int = 0

    def __init__(self, model_name: str = "granite3.3:8b", base_url: str = "http://localhost:11434"):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        self.api_calls = 0

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        """Async generate response using Ollama. This is the primary method for generation."""
        self.api_calls += 1

        # Convert messages to Ollama format
        prompt = self._convert_messages_to_prompt(messages)

        try:
            # Using an async library like httpx would be ideal, but for simplicity,
            # we run the synchronous requests call in a separate thread to avoid blocking the event loop.
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,  # Use the default thread pool executor
                lambda: requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=120
                )
            )

            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()
            return self._create_chat_result(result.get("response", ""))

        except requests.RequestException as e:
            # Handle network-related errors
            return self._create_chat_result(f"Error: Network request failed: {str(e)}")
        except Exception as e:
            # Handle other errors
            return self._create_chat_result(f"Error: An unexpected error occurred: {str(e)}")

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        """
        Synchronous wrapper for _agenerate.
        This is a fallback and should be avoided in async contexts.
        """
        try:
            # Check if an event loop is running.
            loop = asyncio.get_running_loop()
            # If so, schedule the async task and wait for it.
            return asyncio.run_coroutine_threadsafe(self._agenerate(messages, stop, run_manager, **kwargs), loop).result()
        except RuntimeError:
            # If no event loop is running, create a new one.
            return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))


    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to Ollama prompt format"""
        prompt_parts = []

        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            else:
                prompt_parts.append(f"Message: {message.content}")

        return "\n".join(prompt_parts)

    def _create_chat_result(self, content: str) -> ChatResult:
        """Create LangChain ChatResult object"""
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "ollama-async"

class LangGraphAnalytics:
    """Performance analytics for LangGraph implementation"""

    def __init__(self):
        self.metrics = LangGraphMetrics(
            node_execution_times={},
            state_transitions=[],
            total_processing_time=0.0,
            preprocessing_time=0.0,
            api_calls=0,
            error_count=0
        )
        self.start_time = time.time()

    def track_node_execution(self, node_name: str, execution_time: float):
        """Track individual node execution times"""
        if node_name not in self.metrics.node_execution_times:
            self.metrics.node_execution_times[node_name] = []
        self.metrics.node_execution_times[node_name].append(execution_time)

    def track_state_transition(self, from_node: str, to_node: str):
        """Track state transitions between nodes"""
        self.metrics.state_transitions.append({
            'from': from_node,
            'to': to_node,
            'timestamp': datetime.now()
        })

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        total_time = time.time() - self.start_time
        self.metrics.total_processing_time = total_time

        # Calculate average execution times per node
        avg_execution_times = {}
        for node, times in self.metrics.node_execution_times.items():
            avg_execution_times[node] = sum(times) / len(times) if times else 0

        return {
            'total_processing_time': total_time,
            'preprocessing_time': self.metrics.preprocessing_time,
            'node_execution_times': avg_execution_times,
            'state_transitions': len(self.metrics.state_transitions),
            'api_calls': self.metrics.api_calls,
            'error_count': self.metrics.error_count
        }

class ExternalDocumentProcessor:
    """External preprocessing - LangGraph philosophy of developer control"""

    def __init__(self, documents_folder: str = "test_documents"):
        self.documents_folder = documents_folder

    def discover_documents(self) -> List[str]:
        """Discover all markdown files in the documents folder"""
        pattern = os.path.join(self.documents_folder, "*.md")
        return glob.glob(pattern)

    def preprocess_documents(self) -> tuple[List[Dict[str, Any]], float]:
        """
        External preprocessing: Developer controls every step
        This demonstrates LangGraph's philosophy of explicit control
        """
        start_time = time.time()
        processed_docs = []

        # Step 1: Discover documents
        document_paths = self.discover_documents()

        if not document_paths:
            print("❌ No documents found in test_documents folder!")
            return [], 0.0

        print(f"📚 Found {len(document_paths)} documents for preprocessing")

        for doc_path in document_paths:
            try:
                # Step 2: Load and read document
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Step 3: Extract structured metadata
                metadata = self._extract_metadata(content)

                processed_doc = {
                    'source': doc_path,
                    'filename': os.path.basename(doc_path),
                    'title': metadata.get('title', 'Unknown'),
                    'authors': metadata.get('authors', []),
                    'abstract': metadata.get('abstract', ''),
                    'full_content': content,
                    'processing_time': time.time() - start_time
                }

                processed_docs.append(processed_doc)
                print(f"  ✓ Processed: {processed_doc['title']}")

            except Exception as e:
                print(f"❌ Error processing {doc_path}: {str(e)}")
                continue

        preprocessing_time = time.time() - start_time
        print(f"📊 Preprocessing completed in {preprocessing_time:.2f} seconds")

        return processed_docs, preprocessing_time

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from document content with explicit rules"""
        metadata = {}
        lines = content.split('\n')

        # Extract title (first non-empty line, usually with # markdown)
        for line in lines:
            if line.strip():
                metadata['title'] = line.replace('#', '').strip()
                break

        # Extract authors using pattern matching
        authors = []
        for line in lines:
            if any(pattern in line for pattern in ['Author:', 'Authors:', 'Dr.', 'Prof.']):
                author_line = re.sub(r'Authors?:', '', line).strip()
                author_parts = re.split(r'[,;&]', author_line)
                authors.extend([author.strip() for author in author_parts if author.strip()])

        metadata['authors'] = authors

        # Extract abstract
        abstract = ""
        in_abstract = False
        for i, line in enumerate(lines):
            if 'Abstract' in line.lower() and not in_abstract:
                in_abstract = True
                # Capture text on the same line after 'Abstract:'
                abstract_content_on_line = line.lower().split('abstract', 1)[-1].lstrip(': ')
                if abstract_content_on_line:
                    abstract += abstract_content_on_line.strip() + " "
                continue
            elif in_abstract and line.strip():
                if line.startswith('#') or 'Introduction' in line:
                    break
                abstract += line.strip() + " "
        metadata['abstract'] = abstract.strip()
        return metadata

class LangGraphResearchSynthesis:
    """LangGraph-style research synthesis with explicit state management"""

    def __init__(self, ollama_model: str = "granite3.3:8b"):
        self.ollama_model = OllamaLLM(ollama_model)
        self.analytics = LangGraphAnalytics()
        self.doc_processor = ExternalDocumentProcessor()
        self.state_history = []

    def log_state_transition(self, from_step: str, to_step: str, state: ResearchState):
        """Log state transitions for debugging"""
        self.analytics.track_state_transition(from_step, to_step)
        self.state_history.append({
            'from': from_step,
            'to': to_step,
            'timestamp': datetime.now(),
            'state_keys': list(state.keys())
        })

    async def preprocess_documents_node(self, state: ResearchState) -> Dict[str, Any]:
        """External preprocessing node - demonstrates LangGraph's explicit control"""
        print("📋 Node: Document Preprocessing")
        start_time = time.time()

        # External preprocessing with full developer control
        processed_docs, preprocessing_time = self.doc_processor.preprocess_documents()

        execution_time = time.time() - start_time
        self.analytics.track_node_execution("preprocess", execution_time)
        self.analytics.metrics.preprocessing_time = preprocessing_time

        self.log_state_transition("start", "preprocess", state)
        return {
            "processed_docs": processed_docs,
            "current_step": "preprocessing_complete",
            "processing_metrics": {
                "preprocessing_time": preprocessing_time,
                "documents_processed": len(processed_docs),
                "execution_time": execution_time
            }
        }

    async def nimrod_analysis_node(self, state: ResearchState) -> Dict[str, Any]:
        """Nimrod's practical analysis node"""
        print("🔵 Node: Nimrod Analysis (Practical)")
        start_time = time.time()

        prompt = self._build_nimrod_prompt(state["processed_docs"])
        messages = [SystemMessage(content=prompt)]
        response = await self.ollama_model.ainvoke(messages)
        analysis_content = response.content

        analysis = self._structure_nimrod_analysis(analysis_content, state["processed_docs"])

        execution_time = time.time() - start_time
        self.analytics.track_node_execution("nimrod_analysis", execution_time)
        self.analytics.metrics.api_calls += self.ollama_model.api_calls

        self.log_state_transition("preprocess", "nimrod_analysis", state)
        return {
            "nimrod_analysis": analysis,
            "current_step": "nimrod_complete",
            "messages": [AIMessage(content=f"Nimrod: {analysis_content}")]
        }

    async def kalkin_analysis_node(self, state: ResearchState) -> Dict[str, Any]:
        """Kalkin's theoretical analysis node"""
        print("🟢 Node: Kalkin Analysis (Theoretical)")
        start_time = time.time()

        prompt = self._build_kalkin_prompt(state["processed_docs"], state["nimrod_analysis"])
        messages = [SystemMessage(content=prompt)]
        response = await self.ollama_model.ainvoke(messages)
        analysis_content = response.content

        analysis = self._structure_kalkin_analysis(analysis_content, state["processed_docs"])

        execution_time = time.time() - start_time
        self.analytics.track_node_execution("kalkin_analysis", execution_time)
        self.analytics.metrics.api_calls += self.ollama_model.api_calls

        self.log_state_transition("nimrod_analysis", "kalkin_analysis", state)
        return {
            "kalkin_analysis": analysis,
            "current_step": "kalkin_complete",
            "messages": [AIMessage(content=f"Kalkin: {analysis_content}")]
        }

    async def ezzgy_coordination_node(self, state: ResearchState) -> Dict[str, Any]:
        """Ezzgy's coordination and quality control node"""
        print("🟡 Node: Ezzgy Coordination (Quality Control)")
        start_time = time.time()

        prompt = self._build_ezzgy_prompt(state["nimrod_analysis"], state["kalkin_analysis"])
        messages = [SystemMessage(content=prompt)]
        response = await self.ollama_model.ainvoke(messages)
        analysis_content = response.content

        analysis = self._structure_ezzgy_analysis(analysis_content)

        execution_time = time.time() - start_time
        self.analytics.track_node_execution("ezzgy_coordination", execution_time)
        self.analytics.metrics.api_calls += self.ollama_model.api_calls

        self.log_state_transition("kalkin_analysis", "ezzgy_coordination", state)
        return {
            "ezzgy_analysis": analysis,
            "current_step": "ezzgy_complete",
            "messages": [AIMessage(content=f"Ezzgy: {analysis_content}")]
        }

    async def synthesis_node(self, state: ResearchState) -> Dict[str, Any]:
        """Final synthesis node"""
        print("🔴 Node: Final Synthesis")
        start_time = time.time()

        prompt = self._build_synthesis_prompt(state["nimrod_analysis"], state["kalkin_analysis"], state["ezzgy_analysis"])
        messages = [SystemMessage(content=prompt)]
        response = await self.ollama_model.ainvoke(messages)
        synthesis_content = response.content

        synthesis = self._structure_synthesis_result(synthesis_content, state)

        execution_time = time.time() - start_time
        self.analytics.track_node_execution("synthesis", execution_time)
        self.analytics.metrics.api_calls += self.ollama_model.api_calls

        self.log_state_transition("ezzgy_coordination", "synthesis", state)
        return {
            "synthesis_result": synthesis,
            "current_step": "synthesis_complete",
            "messages": [AIMessage(content=f"Final Synthesis: {synthesis_content}")]
        }

    def _build_nimrod_prompt(self, documents: List[Dict[str, Any]]) -> str:
        doc_summaries = [f"Title: {doc['title']}\nAuthors: {', '.join(doc['authors'])}\nAbstract: {doc['abstract']}" for doc in documents]
        return f"You are Nimrod, a practical research analyst. Analyze these papers focusing on concrete data, methodology, and practical implications:\n\n" + "\n\n".join(doc_summaries)

    def _build_kalkin_prompt(self, documents: List[Dict[str, Any]], nimrod_analysis: Dict[str, Any]) -> str:
        doc_summaries = [f"Title: {doc['title']}\nAbstract: {doc['abstract']}" for doc in documents]
        return f"You are Kalkin, a theoretical researcher. Based on Nimrod's practical analysis:\n{nimrod_analysis['analysis']}\n\nNow, analyze these papers from a theoretical perspective, focusing on frameworks and conceptual connections:\n\n" + "\n\n".join(doc_summaries)

    def _build_ezzgy_prompt(self, nimrod_analysis: Dict[str, Any], kalkin_analysis: Dict[str, Any]) -> str:
        return f"You are Ezzgy, a meticulous research coordinator. Given the practical analysis from Nimrod and theoretical analysis from Kalkin, identify gaps, contradictions, and synthesis priorities.\n\nNimrod's Analysis:\n{nimrod_analysis['analysis']}\n\nKalkin's Analysis:\n{kalkin_analysis['analysis']}"

    def _build_synthesis_prompt(self, nimrod_analysis: Dict[str, Any], kalkin_analysis: Dict[str, Any], ezzgy_analysis: Dict[str, Any]) -> str:
        return f"Create a comprehensive research synthesis based on the following analyses:\n\nPractical (Nimrod):\n{nimrod_analysis['analysis']}\n\nTheoretical (Kalkin):\n{kalkin_analysis['analysis']}\n\nCoordination (Ezzgy):\n{ezzgy_analysis['analysis']}\n\nProvide an executive summary, reconcile perspectives, and offer recommendations."

    def _structure_nimrod_analysis(self, response: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"approach": "practical_quantitative", "analysis": response, "documents_analyzed": len(documents), "timestamp": datetime.now()}

    def _structure_kalkin_analysis(self, response: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"approach": "theoretical_conceptual", "analysis": response, "documents_analyzed": len(documents), "timestamp": datetime.now()}

    def _structure_ezzgy_analysis(self, response: str) -> Dict[str, Any]:
        return {"approach": "methodical_coordination", "analysis": response, "timestamp": datetime.now()}

    def _structure_synthesis_result(self, response: str, state: ResearchState) -> Dict[str, Any]:
        return {"final_synthesis": response, "documents_synthesized": len(state["processed_docs"]), "synthesis_timestamp": datetime.now()}

    def create_research_graph(self) -> CompiledStateGraph:
        """Create the LangGraph research synthesis workflow"""
        workflow = StateGraph(ResearchState)

        workflow.add_node("preprocess", self.preprocess_documents_node)
        workflow.add_node("nimrod_analysis", self.nimrod_analysis_node)
        workflow.add_node("kalkin_analysis", self.kalkin_analysis_node)
        workflow.add_node("ezzgy_coordination", self.ezzgy_coordination_node)
        workflow.add_node("synthesis", self.synthesis_node)

        workflow.add_edge(START, "preprocess")
        workflow.add_edge("preprocess", "nimrod_analysis")
        workflow.add_edge("nimrod_analysis", "kalkin_analysis")
        workflow.add_edge("kalkin_analysis", "ezzgy_coordination")
        workflow.add_edge("ezzgy_coordination", "synthesis")
        workflow.add_edge("synthesis", END)

        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    async def run_research_workflow(self) -> Dict[str, Any]:
        """Run the complete LangGraph research workflow asynchronously."""
        print("🚀 Starting LangGraph Research Synthesis...")
        print("="*60)

        os.makedirs("results", exist_ok=True)
        thread_id = str(uuid.uuid4())
        initial_state: ResearchState = {"messages": [], "thread_id": thread_id}

        try:
            graph = self.create_research_graph()
            config = RunnableConfig(configurable={"thread_id": thread_id})

            # Use ainvoke for asynchronous execution
            final_state = await graph.ainvoke(initial_state, config)

            if not final_state.get("processed_docs"):
                print("❌ No documents were processed!")
                return {}

            # Reset API call count for accurate reporting per run
            self.ollama_model.api_calls = 0
            performance_report = self.analytics.generate_performance_report()
            performance_report['api_calls'] = self.analytics.metrics.api_calls

            result = {
                'framework': 'LangGraph',
                'approach': 'State machine with external preprocessing',
                'final_state': final_state,
                'performance_metrics': performance_report,
                'state_transitions': self.state_history,
                'processing_summary': {
                    'total_nodes': 5,
                    'documents_processed': len(final_state.get("processed_docs", [])),
                    'preprocessing_time': performance_report['preprocessing_time'],
                    'total_time': performance_report['total_processing_time']
                }
            }

            print("\n" + "="*60)
            print("📊 LANGGRAPH SYNTHESIS COMPLETE")
            print("="*60)
            print(f"Documents Processed: {len(final_state.get('processed_docs', []))}")
            print(f"Workflow Nodes: 5")
            print(f"Total Processing Time: {performance_report['total_processing_time']:.2f}s")
            print(f"API Calls Made: {performance_report['api_calls']}")

            return result

        except Exception as e:
            print(f"❌ Error in workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            self.analytics.metrics.error_count += 1
            return {}

async def main():
    """Main execution function"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama is not running. Please start Ollama first.")
            return None
    except requests.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {str(e)}")
        return None

    os.makedirs("test_documents", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    synthesizer = LangGraphResearchSynthesis(ollama_model="granite3.3:8b")
    result = await synthesizer.run_research_workflow()

    if result:
        with open('results/langgraph_results.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n📁 Results saved to: results/langgraph_results.json")
        print("✅ LangGraph research synthesis completed successfully!")

    return result

if __name__ == "__main__":
    asyncio.run(main())
