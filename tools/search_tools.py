"""Research tools

This module provides search and content processing utilities for the research agent,
using Chroma vector database for local retrieval and fetching full webpage content."""
import httpx
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from typing_extensions import Annotated
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb

def initial_vector_store():
    """Initialize vector store"""
    client = chromadb.HttpClient(host="192.168.0.162", port=9000, ssl=False)
    embeddings = OllamaEmbeddings(
        base_url="http://192.168.0.162:11434",
        model="embeddinggemma:latest"
    )
    return Chroma(
        collection_name="research",
        client=client,
        embedding_function=embeddings
    )

vector_store = initial_vector_store()

def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown
    
    Args:
        url: URL to fetch
        timeout: request timeout in seconds
    Returns:
        Webpage content as markdown"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"
    
@tool(parse_docstring=True)
def chroma_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
) -> str:
    """Search the local vector database for information on a given query.

    Uses Chroma vector database to retrieve relevant documents, then attempts to fetch
    full webpage content. Falls back to stored snippets if fetching fails.

    Args:
        query: Search query to execute
        max_results: maximum number of results to return (default: 1)

    Returns:
        Formatted search results with full webpage content or snippets
    """
    # retrieve documents from chroma vector database
    search_results = vector_store.similarity_search(query, k=max_results)

    # process each result
    result_texts = []
    for result in search_results:
        snippet = result.page_content
        url = result.metadata.get("url", "Unknown URL")

        # extract title from metadata 
        title = result.metadata.get("title", "Unknown title")

        # try ti fetch the full webpage content
        content = fetch_webpage_content(url)

        # fetching failed -> fall back to the snippet from vector DB
        if content.startswith("Error fetching content"):
            content = f"**[Using cached snippet - full page unavailable]**\n\n{snippet}"

        result_text = f"""## {title}
**URL:** {url}

{content}

---
"""
        result_texts.append(result_text)

    # format final response
    response = f"""🔍 Found {len(result_texts)} result(s) for '{query}':
    
    {chr(10).join(result_texts)}"""

    return response

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"