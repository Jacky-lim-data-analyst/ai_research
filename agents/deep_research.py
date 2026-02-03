"""Research agent - deep research agent with custom tools and prompts 
for conducting web search and rag (local vector search) with strategic thinking and
context management
"""

from datetime import datetime

from config.settings import AppSettings
import os
# from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent

from config.prompts import (
    RESEARCHER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS
)
from tools.search_tools import think_tool, chroma_search

# Limits concurrent research units and iterations
max_concurrent_research_units = 2
max_researcher_iterations = 3

# get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

# create research sub_agent
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [chroma_search, think_tool],
}

# Ollama model
settings = AppSettings()
os.environ["GEMINI_API_KEY"] = settings.gemini_api_key.get_secret_value()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0
)

# create deep agent
agent = create_deep_agent(
    model=model,
    tools=[chroma_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent]
)

# result = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "context": "research context engineering approaches used to build AI agents"
#             }
#         ]
#     }
# )

# file_content = file_data_to_string(result["files"]['/final_report.md'])
