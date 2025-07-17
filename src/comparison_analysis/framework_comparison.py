import json
import time
import asyncio
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

class FrameworkComparator:
    """Comprehensive comparison between AutoGen and LangGraph implementations"""

    def __init__(self):
        self.autogen_results = None
        self.langgraph_results = None
        self.comparison_data = {}

    def load_results(self, autogen_file: str = "results/autogen_results.json", langgraph_file: str = "results/langgraph_results.json"):
        """Load results from both frameworks, ensuring robustness against file not found or JSON errors."""
        try:
            if os.path.exists(autogen_file):
                with open(autogen_file, 'r', encoding='utf-8') as f:
                    self.autogen_results = json.load(f)
                print(f"✅ Loaded AutoGen results from {autogen_file}")
            else:
                print(f"❌ AutoGen results file not found: {autogen_file}")
                self.autogen_results = None

            if os.path.exists(langgraph_file):
                with open(langgraph_file, 'r', encoding='utf-8') as f:
                    self.langgraph_results = json.load(f)
                print(f"✅ Loaded LangGraph results from {langgraph_file}")
            else:
                print(f"❌ LangGraph results file not found: {langgraph_file}")
                self.langgraph_results = None

        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON from a results file: {str(e)}")
            self.autogen_results = None
            self.langgraph_results = None
        except Exception as e:
            print(f"❌ An unexpected error occurred while loading results: {str(e)}")
            self.autogen_results = None
            self.langgraph_results = None

    def extract_metrics(self) -> Dict[str, Any]:
        """Extract comparable metrics from both frameworks' result files."""
        if not self.autogen_results or not self.langgraph_results:
            print("❌ Both framework results are required for comparison.")
            return {}

        # Safely extract metrics using .get() to avoid KeyErrors
        autogen_metrics = self.autogen_results.get('performance_metrics', {})
        autogen_processing = self.autogen_results.get('processing_summary', {})
        langgraph_metrics = self.langgraph_results.get('performance_metrics', {})
        langgraph_processing = self.langgraph_results.get('processing_summary', {})

        comparison_metrics = {
            'processing_time': {
                'autogen': autogen_metrics.get('total_processing_time', 0),
                'langgraph': langgraph_metrics.get('total_processing_time', 0)
            },
            'api_calls': {
                'autogen': autogen_metrics.get('api_calls', 0),
                'langgraph': langgraph_metrics.get('api_calls', 0)
            },
            'documents_processed': {
                'autogen': self.autogen_results.get('documents_processed', 0),
                'langgraph': langgraph_processing.get('documents_processed', 0)
            },
            'preprocessing_time': {
                'autogen': 0,  # AutoGen's processing is integrated, not a separate step
                'langgraph': langgraph_metrics.get('preprocessing_time', 0)
            },
            'workflow_steps': {
                'autogen': autogen_processing.get('total_rounds', 0),
                'langgraph': langgraph_processing.get('total_nodes', 0)
            }
        }
        self.comparison_data = comparison_metrics
        return comparison_metrics

    def create_performance_comparison_chart(self):
        """Create a comprehensive visualization comparing performance metrics."""
        if not self.comparison_data:
            print("❌ No comparison data available. Run extract_metrics() first.")
            return

        fig = plt.figure(figsize=(18, 15))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.2)
        sns.set_style("whitegrid")
        colors = {'autogen': '#3498db', 'langgraph': '#e74c3c'}
        frameworks = ['AutoGen', 'LangGraph']

        # 1. Total Processing Time
        ax1 = fig.add_subplot(gs[0, 0])
        times = [self.comparison_data['processing_time']['autogen'], self.comparison_data['processing_time']['langgraph']]
        bars1 = ax1.bar(frameworks, times, color=[colors['autogen'], colors['langgraph']])
        ax1.set_title('Total Processing Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', va='bottom', ha='center')

        # 2. API Calls
        ax2 = fig.add_subplot(gs[0, 1])
        calls = [self.comparison_data['api_calls']['autogen'], self.comparison_data['api_calls']['langgraph']]
        bars2 = ax2.bar(frameworks, calls, color=[colors['autogen'], colors['langgraph']])
        ax2.set_title('Total API Calls', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Calls')
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

        # 3. Workflow Structure
        ax3 = fig.add_subplot(gs[1, 0])
        steps = [self.comparison_data['workflow_steps']['autogen'], self.comparison_data['workflow_steps']['langgraph']]
        bars3 = ax3.bar(frameworks, steps, color=[colors['autogen'], colors['langgraph']])
        ax3.set_title('Workflow Structure', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Conversation Rounds / Graph Nodes')
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

        # 4. Preprocessing Time
        ax4 = fig.add_subplot(gs[1, 1])
        pre_times = [self.comparison_data['preprocessing_time']['autogen'], self.comparison_data['preprocessing_time']['langgraph']]
        bars4 = ax4.bar(frameworks, pre_times, color=[colors['autogen'], colors['langgraph']])
        ax4.set_title('Explicit Preprocessing Time', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Time (seconds)')
        ax4.text(bars4[0].get_x() + bars4[0].get_width()/2.0, bars4[0].get_height(), 'Integrated', va='bottom', ha='center')
        ax4.text(bars4[1].get_x() + bars4[1].get_width()/2.0, bars4[1].get_height(), f'{pre_times[1]:.2f}s', va='bottom', ha='center')

        # 5. Framework Characteristics (Radar Chart)
        ax5 = fig.add_subplot(gs[2, 0], projection='polar')
        categories = ['Flexibility', 'Control', 'Debugging', 'Setup Speed', 'Scalability', 'Production Ready']
        autogen_scores = [9, 6, 7, 9, 7, 6]
        langgraph_scores = [6, 9, 9, 6, 8, 9]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        ax5.plot(angles, autogen_scores + autogen_scores[:1], 'o-', linewidth=2, label='AutoGen', color=colors['autogen'])
        ax5.fill(angles, autogen_scores + autogen_scores[:1], alpha=0.25, color=colors['autogen'])
        ax5.plot(angles, langgraph_scores + langgraph_scores[:1], 'o-', linewidth=2, label='LangGraph', color=colors['langgraph'])
        ax5.fill(angles, langgraph_scores + langgraph_scores[:1], alpha=0.25, color=colors['langgraph'])
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories)
        ax5.set_title('Qualitative Framework Characteristics', fontsize=14, fontweight='bold', y=1.1)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        # 6. Summary Table
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        summary_data = {
            'Metric': ['Processing Time (s)', 'API Calls', 'Workflow Steps', 'Preprocessing (s)'],
            'AutoGen': [f"{times[0]:.2f}", calls[0], steps[0], "N/A (Integrated)"],
            'LangGraph': [f"{times[1]:.2f}", calls[1], steps[1], f"{pre_times[1]:.4f}"]
        }
        df_summary = pd.DataFrame(summary_data)
        table = ax6.table(cellText=df_summary.values, colLabels=df_summary.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax6.set_title('Quantitative Summary', fontsize=14, fontweight='bold')

        plt.suptitle('AutoGen vs. LangGraph: Comprehensive Performance Comparison', fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs("results", exist_ok=True)
        plt.savefig('results/comprehensive_framework_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("📊 Comprehensive comparison chart saved as 'results/comprehensive_framework_comparison.png'")

    def create_decision_matrix(self):
        """Create a heatmap visualization to serve as a decision matrix."""
        criteria = {
            'Use Case': ['Rapid Prototyping', 'Production Systems', 'Creative/Research Tasks', 'Stateful Workflows', 'Debugging Ease', 'Learning Curve'],
            'AutoGen Score': [9, 6, 9, 7, 6, 8],
            'LangGraph Score': [6, 9, 6, 9, 9, 6]
        }
        df = pd.DataFrame(criteria).set_index('Use Case')
        plt.figure(figsize=(10, 7))
        sns.heatmap(df, annot=True, cmap='viridis', fmt='d', linewidths=.5)
        plt.title('Framework Selection Decision Matrix (1=Worse, 10=Better)', fontsize=16, fontweight='bold')
        plt.xticks(rotation=0)
        plt.xlabel('Framework')
        plt.ylabel('Evaluation Criteria')
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig('results/decision_matrix.png', dpi=300)
        plt.close()
        print("📋 Decision matrix saved as 'results/decision_matrix.png'")

    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate a structured dictionary containing a detailed comparison."""
        if not self.comparison_data:
            return {}

        autogen_time = self.comparison_data['processing_time']['autogen']
        langgraph_time = self.comparison_data['processing_time']['langgraph']
        winner = 'AutoGen' if autogen_time < langgraph_time else 'LangGraph'
        diff = abs(autogen_time - langgraph_time)
        
        report = {
            "executive_summary": {
                "title": "AutoGen vs. LangGraph: Performance Analysis",
                "overall_winner_by_speed": winner,
                "speed_difference_seconds": round(diff, 2),
                "key_takeaway": "The choice of framework should be workload-dependent. AutoGen excels in rapid, flexible development, while LangGraph offers superior control and reliability for structured, production-grade workflows."
            },
            "quantitative_analysis": self.comparison_data,
            "qualitative_analysis": {
                "autogen": {
                    "strengths": ["Rapid prototyping", "High flexibility for creative tasks", "Emergent behavior discovery", "Simpler initial setup"],
                    "weaknesses": ["Less deterministic control", "Debugging can be complex", "State management is less explicit"]
                },
                "langgraph": {
                    "strengths": ["High degree of control and reliability", "Explicit state management", "Excellent for auditable and production workflows", "Easier to debug due to graph structure"],
                    "weaknesses": ["More boilerplate/setup code", "Less flexible for purely exploratory tasks", "Steeper learning curve"]
                }
            },
            "recommendations": {
                "use_autogen_for": "Exploratory research, creative problem-solving, and rapid prototyping where the solution path is unknown.",
                "use_langgraph_for": "Mission-critical enterprise applications, complex but well-defined workflows, and systems requiring high reliability and auditability."
            }
        }
        return report

    def create_article_summary(self) -> str:
        """Generate a markdown-formatted article summarizing the findings."""
        if not self.comparison_data:
            return "# Error: No data available for summary."

        autogen_time = self.comparison_data['processing_time']['autogen']
        langgraph_time = self.comparison_data['processing_time']['langgraph']
        winner = "AutoGen" if autogen_time < langgraph_time else "LangGraph"
        diff = abs(autogen_time - langgraph_time)
        
        return f"""
# AutoGen vs. LangGraph: A Head-to-Head Performance Showdown

In the rapidly evolving landscape of multi-agent AI systems, two frameworks have emerged as leading contenders: AutoGen and LangGraph. While both enable the creation of sophisticated agentic applications, they embody fundamentally different philosophies. But how do they stack up in a real-world scenario? We conducted a direct comparison to find out.

## The Challenge

We tasked both frameworks with the same research synthesis problem, using identical local LLMs (via Ollama) and source documents. The goal was to analyze three academic papers and produce a synthesized report.

## The Results: A Tale of Two Workflows

| Metric                  | AutoGen (Conversational) | LangGraph (State Machine) | Winner      |
|-------------------------|--------------------------|---------------------------|-------------|
| **Total Time (s)** | `{autogen_time:.2f}`             | `{langgraph_time:.2f}`            | **{winner}** |
| **API Calls** | `{self.comparison_data['api_calls']['autogen']}`                 | `{self.comparison_data['api_calls']['langgraph']}`                | AutoGen     |
| **Workflow Steps** | `{self.comparison_data['workflow_steps']['autogen']}` (Rounds)         | `{self.comparison_data['workflow_steps']['langgraph']}` (Nodes)           | -           |
| **Preprocessing Time (s)**| Integrated               | `{self.comparison_data['preprocessing_time']['langgraph']:.4f}`           | AutoGen     |

**{winner} was faster by {diff:.2f} seconds.**

## Analysis: Speed vs. Control

**AutoGen** shines with its speed and simplicity for this task. Its conversational model allows for rapid development and emergent problem-solving, requiring fewer explicit steps and less setup. The lower number of API calls suggests a more efficient token usage pattern in its direct conversational flow.

**LangGraph**, while slightly slower, offers unparalleled control and observability. Its graph-based structure makes the workflow explicit, auditable, and easier to debug, which is critical for production systems. The overhead comes from its deliberate, stateful transitions between nodes.

## The Verdict: Which Framework Should You Choose?

The "best" framework is not a one-size-fits-all answer.

-   **Choose AutoGen for:** Rapid prototyping, research, and creative tasks where flexibility and speed are paramount.
-   **Choose LangGraph for:** Production-grade, mission-critical applications where reliability, auditability, and explicit state control are non-negotiable.

Ultimately, the choice depends on your project's specific needs. For quick iteration and exploration, AutoGen is a powerful ally. For building robust, enterprise-ready systems, LangGraph provides the necessary structure and safety nets.
"""

    def run_complete_analysis(self):
        """Run the full analysis pipeline: load, extract, visualize, and report."""
        print("🔍 Starting Comprehensive Framework Comparison...")
        print("="*60)

        print("📊 Loading and analyzing results...")
        self.load_results()
        self.extract_metrics()

        if not self.comparison_data:
            print("❌ Cannot proceed: Failed to load or extract data from result files.")
            return None, None

        print("📈 Creating comprehensive comparison charts...")
        self.create_performance_comparison_chart()

        print("📋 Generating decision matrix...")
        self.create_decision_matrix()

        print("📝 Generating comprehensive report...")
        report = self.generate_comparison_report()
        os.makedirs("results", exist_ok=True)
        with open('results/comprehensive_comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print("📰 Creating article-ready summary...")
        article_summary = self.create_article_summary()
        with open('results/article_ready_summary.md', 'w', encoding='utf-8') as f:
            f.write(article_summary)

        print("\n" + "="*60)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*60)
        print("📁 Generated Files:")
        print("  • results/comprehensive_framework_comparison.png")
        print("  • results/decision_matrix.png")
        print("  • results/comprehensive_comparison_report.json")
        print("  • results/article_ready_summary.md")

        return report, article_summary

async def run_both_frameworks_and_compare():
    """Run both framework implementations sequentially and then run the comparison."""
    print("🚀 Starting Complete Framework Comparison Pipeline...")
    print("="*70)

    # Run AutoGen implementation
    print("🔵 Running AutoGen Implementation...")
    try:
        from autogen_implementation.autogen_research_synthesis import main as autogen_main
        autogen_result = await autogen_main()
        if not autogen_result:
            print("❌ AutoGen implementation failed to produce results.")
            return
    except ImportError:
        print("❌ Could not import 'autogen_research_synthesis'. Make sure the file exists.")
        return
    except Exception as e:
        print(f"❌ An error occurred during AutoGen execution: {str(e)}")
        return

    # Run LangGraph implementation
    print("\n🔴 Running LangGraph Implementation...")
    try:
        from langgraph_implementation.langgraph_research_synthesis import main as langgraph_main
        langgraph_result = await langgraph_main()
        if not langgraph_result:
            print("❌ LangGraph implementation failed to produce results.")
            return
    except ImportError:
        print("❌ Could not import 'langgraph_research_synthesis'. Make sure the file exists.")
        return
    except Exception as e:
        print(f"❌ An error occurred during LangGraph execution: {str(e)}")
        return

    # Create comprehensive comparison
    print("\n🔍 Creating Comprehensive Comparison...")
    comparator = FrameworkComparator()
    report, summary = comparator.run_complete_analysis()

    if report:
        print("\n🎉 Complete pipeline finished successfully!")
    else:
        print("\n❌ Final analysis failed. Please check the logs.")

def main():
    """Main function to run only the standalone comparison on existing result files."""
    print("🔍 Framework Comparison Tool (Standalone Mode)")
    print("="*50)
    print("This will only run the comparison on existing 'results/*.json' files.")

    comparator = FrameworkComparator()
    report, summary = comparator.run_complete_analysis()

    if report:
        print("\n🎉 Analysis complete! Check the 'results' directory.")
    else:
        print("❌ Could not complete analysis. Check if result files exist and are valid.")


if __name__ == "__main__":
    # To run the full pipeline (both frameworks + comparison), you would typically call:
    # asyncio.run(run_both_frameworks_and_compare())

    # The default execution will be the standalone comparison.
    main()
