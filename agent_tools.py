# agent_tools.py
from langchain.agents import Tool

def get_agent_tools(qa_chain):
    tools = [
        Tool(name="PDF_QA", func=qa_chain.run, description="Answer questions based on uploaded PDF"),
        # Add more tools later (summarization, translation, etc.)
    ]
    return tools
