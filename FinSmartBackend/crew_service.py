"""Crew service wrapper for stock analysis"""
import sys
import os
import re
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Add Agent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Agent'))

from agents import StockAnalysisAgents
from tasks import StockAnalysisTasks
from crewai import Crew, Process


# Removed fix_markdown_formatting as it corrupted LLM tables


class CrewService:
    """Service for executing stock analysis crew"""
    
    def __init__(self):
        """Initialize agents and tasks"""
        self.agents = StockAnalysisAgents()
        self.tasks = StockAnalysisTasks()
        
        # Initialize agents
        self.financial_analyst = self.agents.financial_analyst()
        self.research_analyst = self.agents.research_analyst()
        self.investment_advisor = self.agents.investment_advisor()
    
    async def analyze_stock(self, company: str) -> dict:
        """
        Run stock analysis crew
        """
        try:
            # Create tasks
            research_task = self.tasks.research(self.research_analyst)
            financial_task = self.tasks.financial_analysis(self.financial_analyst)
            filings_task = self.tasks.filings_analysis(self.financial_analyst)
            recommend_task = self.tasks.recommend(self.investment_advisor)
            
            # Create crew
            crew = Crew(
                agents=[
                    self.financial_analyst,
                    self.research_analyst,
                    self.investment_advisor
                ],
                tasks=[
                    research_task,
                    financial_task,
                    filings_task,
                    recommend_task
                ],
                process=Process.sequential,
                memory=False,
                cache=False,
                max_rpm=100,
                share_crew=True,
                full_output=True,
                max_iter=15
            )
            
            # Execute crew in a thread pool (kickoff is synchronous/blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: crew.kickoff(inputs={"company": company})
            )
            
            # Build the full report from ALL task outputs recursively
            analysis_text = ""
            section_names = [
                "Market Research & News Analysis",
                "Financial Analysis & Metrics",
                "SEC Filings & Earnings Analysis",
                "Investment Recommendation"
            ]
            
            if hasattr(result, 'tasks_output') and result.tasks_output:
                sections = []
                for i, task_output in enumerate(result.tasks_output):
                    task_raw = ""
                    if hasattr(task_output, 'raw') and task_output.raw:
                        task_raw = task_output.raw
                    elif hasattr(task_output, 'output') and task_output.output:
                        task_raw = task_output.output
                    else:
                        task_raw = str(task_output)
                    
                    if task_raw and len(task_raw.strip()) > 50:
                        header = section_names[i] if i < len(section_names) else f"Section {i+1}"
                        sections.append(f"## {header}\n\n{task_raw.strip()}")
                
                if sections:
                    analysis_text = f"# Investment Report: {company}\n\n" + "\n\n---\n\n".join(sections)
            
            # Fallback to single .raw output if tasks_output didn't work
            if not analysis_text or len(analysis_text) < 500:
                if hasattr(result, 'raw') and result.raw:
                    analysis_text = result.raw
                else:
                    analysis_text = str(result)
                
            logger.info(f"Final analysis length: {len(analysis_text)} chars")
            
            return {
                "status": "success",
                "company": company,
                "analysis": analysis_text,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Stock analysis failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "company": company,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
