"""Research tools

This module provides search and content processing utilities for the research agent,
using Chroma vector database for local retrieval and fetching full webpage content."""
import httpx
import re
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from typing import Optional
from typing_extensions import Annotated
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb
from tavily import TavilyClient
from config.settings import AppSettings

class TavilyWebExtractor:
    """Thin wrapper around Tavily's `extract` API that returns a *clean, LLM-ready string*.
    Responsibilities:
    - Call Tavily extract
    - Pull `results[0].raw_content`
    - Normalize escapes and whitespace
    - Remove obvious navigation / boilerplate junk"""
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Tavily API key not found")
        self.client = TavilyClient(api_key=api_key)
    
    def extract(self, url: str) -> str:
        """
        Extract and clean web content from a URL.
        Returns:
        Cleaned plain text string suitable for LLM input.
        Raises:
        RuntimeError: if extraction fails or no content is returned.
        """
        response = self.client.extract(url)

        try:
            raw_content: Optional[str] = response["results"][0].get("raw_content")
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("Invalid Tavily response format")
        
        if not raw_content:
            raise RuntimeError("No raw content returned by Tavily")
        
        return self._clean_text(raw_content)
    
    def _clean_text(self, text: str) -> str:
        """
        Heuristic cleanup pass.
        Tavily already strips HTML; this removes navigation noise
        and makes the text LLM-friendly.
        """
        # 1. unescape common sequences
        text = text.replace("\\n", "\n").replace("\\t", " ")

        # 2. Drop obvious wiki / site chrome lines
        blacklist_patterns = [
            r"^Jump to content$",
            r"^Main menu$",
            r"^Search$",
            r"^Donate$",
            r"^Create account$",
            r"^Log in$",
            r"^Personal tools$",
            r"^Toggle the table of contents$",
            r"^View source$",
            r"^View history$",
            r"^From Wikipedia, the free encyclopedia$",
            r"^\d+ languages$",
        ]

        cleaned_lines: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            if any(re.match(pat, line) for pat in blacklist_patterns):
                continue

            cleaned_lines.append(line)

        # 3 rejoin and normalize whitespace
        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

        return cleaned_text.strip()

def initialize_vector_store(collection_name: str):
    """Initialize vector store"""
    client = chromadb.HttpClient(host="192.168.0.162", port=9000, ssl=False)
    embeddings = OllamaEmbeddings(
        base_url="http://192.168.0.162:11434",
        model="embeddinggemma:latest"
    )
    return Chroma(
        collection_name=collection_name,
        client=client,
        embedding_function=embeddings
    )

vector_store = initialize_vector_store(collection_name="artemis_space")

try: 
    settings = AppSettings()
    tavily_extractor = TavilyWebExtractor(api_key=settings.tavily_api_key.get_secret_value())
except (ValueError, Exception):
    tavily_extractor = None

def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown
    Tries handlers in order:
    1. Tavily extract API (if available)
    2. httpx.get() with markdownify

    Args:
        url: URL to fetch
        timeout: request timeout in seconds
    Returns:
        Webpage content as markdown"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    # handler 1: try Tavily extract first
    if tavily_extractor:
        try:
            content = tavily_extractor.extract(url)
            return content
        except RuntimeError as e:
            print(f"Tavily extraction failed for {url}: {e}")
        except Exception as e:
            print(f"Unexpected Tavily error for {url}: {e}")

    # handler 2:
    # connection and read timeout
    timeout_cfg = httpx.Timeout(5.0, read=timeout)

    try:
        response = httpx.get(url, headers=headers, timeout=timeout_cfg)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} at {url}")
        return f"HTTP error: {e.response.status_code} at {url}"
    except httpx.RequestError as e:
        print(f"Network error fetching {url}: {e}")
        return f"Network error fetching {url}: {e}"
    except Exception as e:
        return f"Error fetching content from {url}: {e}"
    
    html = response.text
    return markdownify(
        html, 
        strip=['style'],
        heading_style="atx"
    )
    
@tool(parse_docstring=True)
def chroma_search(
    query: str,
    # vector_store: Annotated[Chroma, InjectedToolArg] = vector_store,
    max_results: Annotated[int, InjectedToolArg] = 10,
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
    retrieval_res = vector_store.similarity_search_with_score(query, k=max_results)

    if not retrieval_res:
        return f"No retrieval data found for query: '{query}'"
    
    # get the most similar result (first one, lowest score). Chroma returns distance measures
    top_doc, _ = retrieval_res[0]
    top_url = top_doc.metadata.get("url", "")
    top_title = top_doc.metadata.get("title", "Unknown title")

    content = None
    fetch_success = False

    # try to fetch the full webpage content of the most similar result
    if top_url:
        content = fetch_webpage_content(top_url)
        # check if fetching was successful
        if not (content.startswith("Error fetching content") or 
                content.startswith("HTTP error") or
                content.startswith("Network error")):
            fetch_success = True

    if not fetch_success:
        snippet_texts = []
        for doc, _ in retrieval_res:
            snippet = doc.page_content
            snippet_texts.append(snippet)

        content = "\n\n".join(snippet_texts)

    # process each result
    if fetch_success:
        # format final response
        response = f"""🔍 Found {len(retrieval_res)} result(s) for '{query}':
## {top_title}
**URL:** {top_url}

{content}
"""
    else: 
        response = f"""🔍 Found {len(retrieval_res)} result(s) for '{query}':

{content}"""

    return response

# @tool(parse_docstring=True)
# def think_tool(reflection: str) -> str:
#     """Tool for strategic reflection on research progress and decision-making.

#     Use this tool after each search to analyze results and plan next steps systematically.
#     This creates a deliberate pause in the research workflow for quality decision-making.

#     When to use:
#     - After receiving search results: What key information did I find?
#     - Before deciding next steps: Do I have enough to answer comprehensively?
#     - When assessing research gaps: What specific information am I still missing?
#     - Before concluding research: Can I provide a complete answer now?

#     Reflection should address:
#     1. Analysis of current findings - What concrete information have I gathered?
#     2. Gap assessment - What crucial information is still missing?
#     3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
#     4. Strategic decision - Should I continue searching or provide my answer?

#     Args:
#         reflection: Your detailed reflection on research progress, findings, gaps, and next steps

#     Returns:
#         Confirmation that reflection was recorded for decision-making
#     """
#     return f"Reflection recorded: {reflection}"
