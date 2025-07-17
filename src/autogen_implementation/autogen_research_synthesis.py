import asyncio
import json
import time
import os
import glob
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Ollama integration
import requests

@dataclass
class ConversationMetrics:
    """Metrics tracking for AutoGen conversations"""
    speaker_turns: Dict[str, int]
    topic_transitions: List[Dict[str, Any]]
    response_times: List[float]
    token_usage: Dict[str, int]
    api_calls: int
    processing_time: float
    memory_usage: float
    emergent_insights: List[str]
    collaboration_patterns: List[str]

class ConversationAnalytics:
    """Real-time conversation monitoring and analytics"""
    
    def __init__(self):
        self.metrics = ConversationMetrics(
            speaker_turns={},
            topic_transitions=[],
            response_times=[],
            token_usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            api_calls=0,
            processing_time=0.0,
            memory_usage=0.0,
            emergent_insights=[],
            collaboration_patterns=[]
        )
        self.start_time = time.time()
        self.conversation_history = []
        
    def track_speaker_turn(self, speaker: str, message: str, response_time: float):
        """Track speaker selection and timing"""
        self.metrics.speaker_turns[speaker] = self.metrics.speaker_turns.get(speaker, 0) + 1
        self.metrics.response_times.append(response_time)
        self.conversation_history.append({
            'speaker': speaker,
            'message': message,
            'timestamp': datetime.now(),
            'response_time': response_time
        })
        
    def analyze_collaboration_patterns(self):
        """Analyze how agents build on each other's ideas"""
        patterns = []
        for i in range(1, len(self.conversation_history)):
            current = self.conversation_history[i]
            previous = self.conversation_history[i-1]
            
            if current['speaker'] != previous['speaker']:
                if any(word in current['message'].lower() for word in ['building on', 'agree', 'adding to']):
                    patterns.append(f"{current['speaker']} built on {previous['speaker']}'s idea")
                elif any(word in current['message'].lower() for word in ['however', 'but', 'disagree']):
                    patterns.append(f"{current['speaker']} challenged {previous['speaker']}'s point")
                    
        self.metrics.collaboration_patterns = patterns
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        total_time = time.time() - self.start_time
        
        return {
            'total_processing_time': total_time,
            'average_response_time': sum(self.metrics.response_times) / len(self.metrics.response_times) if self.metrics.response_times else 0,
            'speaker_distribution': self.metrics.speaker_turns,
            'topic_transitions': len(self.metrics.topic_transitions),
            'emergent_insights_count': len(self.metrics.emergent_insights),
            'collaboration_patterns_count': len(self.metrics.collaboration_patterns),
            'api_calls': self.metrics.api_calls
        }

class DocumentProcessor:
    """Built-in document processing - AutoGen philosophy"""
    
    def __init__(self, documents_folder: str = "test_documents"):
        self.documents_folder = documents_folder
        
    def discover_documents(self) -> List[str]:
        """Automatically discover documents in the folder"""
        pattern = os.path.join(self.documents_folder, "*.md")
        return glob.glob(pattern)
        
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract metadata from content
            lines = content.split('\n')
            title = lines[0].replace('#', '').strip() if lines else os.path.basename(file_path)
            
            # Extract authors
            authors = []
            for line in lines[:10]:  # Check first 10 lines
                if 'Author' in line or 'Dr.' in line:
                    authors.append(line.strip())
                    
            # Extract abstract
            abstract = ""
            for i, line in enumerate(lines):
                if 'Abstract' in line and i < len(lines) - 1:
                    abstract = lines[i+1].strip()
                    break
                    
            return {
                'file_path': file_path,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'content': content,
                'length': len(content),
                'processing_time': time.time()
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

class OllamaAgent:
    """Simple Ollama agent for direct interaction"""
    
    def __init__(self, name: str, system_message: str, model: str = "granite3.3:8b"):
        self.name = name
        self.system_message = system_message
        self.model = model
        self.base_url = "http://localhost:11434"
        self.api_calls = 0
        
    async def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama"""
        self.api_calls += 1
        
        # Combine system message with prompt
        full_prompt = f"System: {self.system_message}\n\nHuman: {prompt}\n\nAssistant:"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"

class SimpleAutoGenResearchTeam:
    """Simplified AutoGen research team with direct agent interaction"""
    
    def __init__(self, ollama_model: str = "granite3.3:8b"):
        self.ollama_model = ollama_model
        self.analytics = ConversationAnalytics()
        self.doc_processor = DocumentProcessor()
        
        # Create agents with distinct personalities
        self.agents = self._create_agents()
        
    def _create_agents(self) -> List[OllamaAgent]:
        """Create the three research agents"""
        
        # Nimrod - The Practical Analyst
        nimrod = OllamaAgent(
            name="Nimrod",
            system_message="""You are Nimrod, a practical research analyst who focuses on concrete data and actionable insights.
            Your approach: Extract hard facts, numbers, and measurable outcomes. Ask "So what?" - what does this mean in practice?
            You're straightforward, no-nonsense, and keep theoretical discussions grounded in reality.
            When analyzing papers, focus on quantitative findings, methodology credibility, and practical applications.
            Always provide specific examples and concrete recommendations. Be concise but thorough.""",
            model=self.ollama_model
        )
        
        # Kalkin - The Theoretical Synthesizer  
        kalkin = OllamaAgent(
            name="Kalkin",
            system_message="""You are Kalkin, a theoretical researcher who excels at finding patterns and connections.
            Your approach: Look for conceptual frameworks, identify philosophical implications, and connect ideas across domains.
            You're academically rigorous and nuanced, helping others see the bigger theoretical picture.
            When analyzing papers, focus on underlying assumptions, theoretical soundness, and conceptual innovations.
            Always explore the deeper implications and broader context of findings. Be thoughtful and analytical.""",
            model=self.ollama_model
        )
        
        # Ezzgy - The Methodical Documenter
        ezzgy = OllamaAgent(
            name="Ezzgy",
            system_message="""You are Ezzgy, a meticulous research coordinator ensuring comprehensive analysis.
            Your role: Track findings, identify gaps, ensure balanced representation, and guide productive synthesis.
            You're the glue that holds the research team together, maintaining conversation flow and quality.
            When participating, summarize key points, ask clarifying questions, and create structured outputs.
            Always ensure all perspectives are considered and nothing important is overlooked. Be organized and systematic.""",
            model=self.ollama_model
        )
        
        return [nimrod, kalkin, ezzgy]
        
    async def run_research_synthesis(self) -> Dict[str, Any]:
        """Run the complete research synthesis with simplified approach"""
        
        print("🚀 Starting AutoGen Research Synthesis...")
        print("="*60)
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Discover and process documents
        print("📄 Discovering documents...")
        document_paths = self.doc_processor.discover_documents()
        
        if not document_paths:
            print("❌ No documents found in test_documents folder!")
            return {}
            
        print(f"📚 Found {len(document_paths)} documents:")
        for path in document_paths:
            print(f"  • {os.path.basename(path)}")
            
        # Process documents
        print("\n📖 Processing documents...")
        documents = []
        for path in document_paths:
            doc = self.doc_processor.process_document(path)
            if doc:
                documents.append(doc)
                print(f"  ✓ {doc['title']}")
                
        if not documents:
            print("❌ No documents could be processed!")
            return {}
            
        # Create research task
        doc_summaries = []
        for doc in documents:
            doc_summaries.append(f"Title: {doc['title']}\nAuthors: {', '.join(doc['authors'])}\nAbstract: {doc['abstract']}")
        
        research_context = f"""
        Research papers to analyze:
        
        {chr(10).join(doc_summaries)}
        
        Task: Conduct a comprehensive research synthesis by:
        1. Extracting and analyzing key findings from each paper
        2. Identifying connections, contradictions, and complementary insights
        3. Synthesizing a comprehensive view of the research landscape
        4. Providing recommendations and conclusions
        """
        
        print(f"\n🤖 Starting conversation with {len(self.agents)} agents...")
        print("="*60)
        
        # Run simplified conversation flow
        try:
            conversation_results = []
            
            # Round 1: Nimrod analyzes practical aspects
            print("🔵 Nimrod analyzing practical aspects...")
            start_time = time.time()
            
            nimrod_prompt = f"{research_context}\n\nAs Nimrod, focus on the practical, quantitative aspects of these papers. What are the concrete findings and actionable insights?"
            nimrod_response = await self.agents[0].generate_response(nimrod_prompt)
            nimrod_time = time.time() - start_time
            
            conversation_results.append({
                'speaker': 'Nimrod',
                'content': nimrod_response,
                'response_time': nimrod_time
            })
            
            self.analytics.track_speaker_turn('Nimrod', nimrod_response, nimrod_time)
            print(f"  ✓ Nimrod completed analysis ({nimrod_time:.2f}s)")
            
            # Round 2: Kalkin synthesizes theoretical connections
            print("🟢 Kalkin exploring theoretical connections...")
            start_time = time.time()
            
            kalkin_prompt = f"{research_context}\n\nBased on Nimrod's practical analysis:\n{nimrod_response}\n\nAs Kalkin, focus on the theoretical frameworks and conceptual connections. What are the deeper implications and patterns?"
            kalkin_response = await self.agents[1].generate_response(kalkin_prompt)
            kalkin_time = time.time() - start_time
            
            conversation_results.append({
                'speaker': 'Kalkin',
                'content': kalkin_response,
                'response_time': kalkin_time
            })
            
            self.analytics.track_speaker_turn('Kalkin', kalkin_response, kalkin_time)
            print(f"  ✓ Kalkin completed analysis ({kalkin_time:.2f}s)")
            
            # Round 3: Ezzgy coordinates and synthesizes
            print("🟡 Ezzgy coordinating synthesis...")
            start_time = time.time()
            
            ezzgy_prompt = f"{research_context}\n\nBased on the team's analysis:\n\nNimrod (Practical): {nimrod_response}\n\nKalkin (Theoretical): {kalkin_response}\n\nAs Ezzgy, coordinate these perspectives into a comprehensive synthesis. What are the key findings, gaps, and recommendations?"
            ezzgy_response = await self.agents[2].generate_response(ezzgy_prompt)
            ezzgy_time = time.time() - start_time
            
            conversation_results.append({
                'speaker': 'Ezzgy',
                'content': ezzgy_response,
                'response_time': ezzgy_time
            })
            
            self.analytics.track_speaker_turn('Ezzgy', ezzgy_response, ezzgy_time)
            print(f"  ✓ Ezzgy completed synthesis ({ezzgy_time:.2f}s)")
            
            # Track total API calls
            total_api_calls = sum(agent.api_calls for agent in self.agents)
            self.analytics.metrics.api_calls = total_api_calls
            
            # Analyze collaboration patterns
            self.analytics.analyze_collaboration_patterns()
            
            # Generate performance report
            performance_report = self.analytics.generate_performance_report()
            
            # Create final synthesis
            final_synthesis = f"""
            RESEARCH SYNTHESIS RESULTS:
            
            NIMROD'S PRACTICAL ANALYSIS:
            {nimrod_response}
            
            KALKIN'S THEORETICAL SYNTHESIS:
            {kalkin_response}
            
            EZZGY'S COORDINATED SYNTHESIS:
            {ezzgy_response}
            """
            
            synthesis_result = {
                'framework': 'AutoGen',
                'approach': 'Conversational with built-in document processing',
                'documents_processed': len(documents),
                'conversation_history': self.analytics.conversation_history,
                'performance_metrics': performance_report,
                'collaboration_patterns': self.analytics.metrics.collaboration_patterns,
                'conversation_results': conversation_results,
                'processing_summary': {
                    'total_messages': len(conversation_results),
                    'unique_speakers': len(set(result['speaker'] for result in conversation_results)),
                    'avg_response_time': sum(result['response_time'] for result in conversation_results) / len(conversation_results),
                    'total_rounds': len(conversation_results)
                },
                'final_result': final_synthesis
            }
            
            print("\n" + "="*60)
            print("📊 AUTOGEN SYNTHESIS COMPLETE")
            print("="*60)
            print(f"Documents Processed: {len(documents)}")
            print(f"Conversation Rounds: {len(conversation_results)}")
            print(f"Total API Calls: {total_api_calls}")
            print(f"Average Response Time: {sum(result['response_time'] for result in conversation_results) / len(conversation_results):.2f}s")
            print(f"Total Processing Time: {performance_report['total_processing_time']:.2f}s")
            
            return synthesis_result
            
        except Exception as e:
            print(f"❌ Error during synthesis: {str(e)}")
            return {}

async def main():
    """Main execution function"""
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("❌ Ollama is not running. Please start Ollama first.")
            print("   Run: ollama serve")
            return None
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {str(e)}")
        print("   Please ensure Ollama is installed and running")
        return None
        
    # Create test_documents folder if it doesn't exist
    os.makedirs("test_documents", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize research team
    research_team = SimpleAutoGenResearchTeam(ollama_model="granite3.3:8b")
    
    # Run synthesis
    result = await research_team.run_research_synthesis()
    
    if result:
        # Save results
        with open('results/autogen_results.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: results/autogen_results.json")
        print("✅ AutoGen research synthesis completed successfully!")
        
    return result

if __name__ == "__main__":
    result = asyncio.run(main())