
# realtime_search_agent.py
# A clean, simplified module for other Python programs to import and use.
import re
from typing import Any, Dict, List, Tuple
from langchain_tavily import TavilySearch
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain.agents import create_agent 

# Set your API keys
import os
os.environ["TAVILY_API_KEY"] = "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"



class RealtimeSearchAgent:
    """
    A minimal, production-ready wrapper that provides:
      - Tavily web search tool
      - Your LLM initialized once
      - A formatted answer with detected sources
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 8000,
        max_results: int = 3,
        system_prompt: str = (
            "You are a helpful assistant with real-time web access. "
            "Use the search tool to gather current information. "
            "Always include URLs of the sources you use."
        )
    ):
        # Environment key validation
        if not os.getenv("TAVILY_API_KEY"):
            raise EnvironmentError("Missing TAVILY_API_KEY environment variable.")

        # Tool
        self.search_tool = TavilySearch(max_results=max_results)

        # LLM initialization
        self.llm = init_llm(
            model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Agent
        self.agent = create_agent(
            model=self.llm,
            tools=[self.search_tool],
            system_prompt=system_prompt,
        )

    def ask(self, query: str) -> Tuple[str, List[str]]:
        """
        Execute the agent and return:
            (formatted_answer, list_of_detected_urls)
        """

        response = self.agent.invoke(
            {
                "messages": [
                    ("user", query)
                ]
            }
        )

        # Extract final LLM reply
        msg = response["messages"][-1]
        content = msg.content

        # Convert block format to text if needed
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )

        # Extract sources by regex
        urls = re.findall(r"https?://[^\s)>\]]+", content)
        urls = list(dict.fromkeys(urls))  # dedupe while preserving order

        return content, urls

