# web_tools.py
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

def get_web_tools(num_results: int = 5):
    """
    Returns two LangChain Tools:
      - WEB_SEARCH_QUICK: one-shot, most relevant snippet
      - WEB_SEARCH_RESULTS: top-N results (titles + snippets)
    """
    # Under the hood these are BaseTool subclasses; we expose them as Tool(func=...).
    ddg_quick = DuckDuckGoSearchRun()                   # .run(query) -> str
    ddg_results = DuckDuckGoSearchResults(num_results=num_results)  # .run(query) -> str

    # define web search tool
    WEB_SEARCH_QUICK = Tool(
        name="WEB_SEARCH_QUICK",
        func=ddg_quick.run,
        description=(
            "Quick web lookup for up-to-date facts. "
            "Use when the answer is not in patient docs/helpbook or needs current info."
        ),
    )

    # Web search top k results 
    WEB_SEARCH_RESULTS = Tool(
        name="WEB_SEARCH_RESULTS",
        func=ddg_results.run,
        description=(
            f"Web search returning top {num_results} results with titles/snippets for cross-checking."
        ),
    )

    return [WEB_SEARCH_QUICK, WEB_SEARCH_RESULTS]
