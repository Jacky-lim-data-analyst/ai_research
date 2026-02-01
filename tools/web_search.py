"""Web search providers"""

from typing import List, Dict, Any, Protocol, Type
import json
import requests

from config.settings import AppSettings

# langchain web search integrations
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, SearxSearchWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from tavily import TavilyClient

# --- starndardize output format ---
# all providers will return a list of dictionaries with these keys:
# {
#   "title": str,
#   "content": str,
#   "url": str
#   "source": str   # name of web search provider
# }
class WebSearchProvider(Protocol):
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]: ...

# ------
# Duckduckgo search
# ------
class DuckDuckGoProvider:
    def __init__(self, **kwargs):
        # configure wrapper based on initialization args
        self.wrapper = DuckDuckGoSearchAPIWrapper(
            source=kwargs.get('source', 'text')
        )

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        num_results = kwargs.get("max_results", 5)
        searcher = DuckDuckGoSearchResults(api_wrapper=self.wrapper, output_format='json', num_results=num_results)

        results = searcher.invoke(query)

        try:
            data_dict_list = json.loads(results)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON from ddgs: {e}. Returning empty list")
            return []
        except Exception as e:
            print(f"Error in loading json: {e}. Returning empty list")
            return []
        
        if not data_dict_list or not data_dict_list[0]:
            return []
        
        return [
            {
                "title": r.get("title"),
                "url": r.get("link"),
                "content": r.get("snippet"),
                "source": "duckduckgo"
            } for r in data_dict_list
        ]
    
# -------
# SearXNG web search provider
# -------
class SearxNGProvider:
    def __init__(self, host: str = "http://localhost:8888"):
        self.wrapper = SearxSearchWrapper(searx_host=host)

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        num_results = kwargs.get("max_results", 5)
        categories = kwargs.get("categories", ["general"])

        # get results
        raw_results = self.wrapper.results(
            query,
            num_results=num_results,
            categories=categories,
        )

        if not raw_results:
            return []
        
        return [
            {
                "title": r.get("title"),
                "url": r.get("link"),
                "content": r.get("snippet"),
                "source": "searxng"
            } for r in raw_results
        ]
    
# --------
# whoogle search
# --------
class WhoogleProvider:
    def __init__(self, base_url: str = "http://192.168.0.162:5000/search"):
        self.base_url = base_url

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        params = {
            "q": query,
            "format": "json",
            "num": kwargs.get("max_results", 5)
        }
        headers = {"Accept": "application/json"}

        try:
            response = requests.get(self.base_url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                raw_results = data.get("results", [])

                if not raw_results:
                    return []
                
                return [
                    {
                        "title": r.get("title"),
                        "url": r.get("href"),
                        "content": r.get("content"),
                        "source": "whoogle"
                    } for r in raw_results
                ]
            else:
                print(f"Whoogle error: {response.status_code}")
                return []
            
        except Exception as ex:
            print(f"Whoogle error - {response.status_code}: {str(ex)}")
            return []

# -----
# Tavily search
# -----  
class TavilyProvider:
    def __init__(self, api_key: str | None = None):
        settings = AppSettings()
        self.api_key = api_key or settings.tavily_api_key.get_secret_value()
        if not self.api_key:
            raise EnvironmentError("TAVILY API KEY not found or provided")
        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        search_depth = kwargs.get('search_depth', 'basic')
        topic = kwargs.get('topic', 'general')
        max_results = kwargs.get('max_results', 5)

        response = self.client.search(
            query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            include_answer=False
        )

        raw_results = response.get("results", [])

        if not raw_results:
            return []

        return [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
                "source": "tavily"
            } for r in raw_results
        ]
    
def get_ws_provider(provider_name: str, **config) -> WebSearchProvider:
    """Factory method for web search providers"""
    providers: Dict[str, Type[WebSearchProvider]] = {
        "duckduckgo": DuckDuckGoProvider,
        "searxng": SearxNGProvider,
        "tavily": TavilyProvider,
        "whoogle": WhoogleProvider
    }

    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        print(f"Warning: provided {provider_name} is not available")
        print("Roll back to ddgs web search")
        return DuckDuckGoProvider(**config)
    
    return provider_class(**config)
