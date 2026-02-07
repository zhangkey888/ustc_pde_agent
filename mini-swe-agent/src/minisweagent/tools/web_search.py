# # src/minisweagent/tools/web_search.py
# import logging

# try:
#     from duckduckgo_search import DDGS
# except ImportError:
#     DDGS = None

# logger = logging.getLogger(__name__)

# class SearchTool:
#     def __init__(self):
#         if DDGS is None:
#             logger.warning("duckduckgo-search not installed. Search tool will not work.")

#     def run(self, query: str, max_results: int = 5) -> str:
#         """执行搜索并返回格式化的字符串给 Agent"""
#         if DDGS is None:
#             return "Error: `duckduckgo-search` library not found. Please run `pip install duckduckgo-search`."
        
#         try:
#             # 执行搜索
#             results = list(DDGS().text(query, max_results=max_results))
#             if not results:
#                 return f"No web results found for: {query}"

#             # 格式化输出
#             output = [f"Search Results for '{query}':"]
#             for i, res in enumerate(results, 1):
#                 title = res.get('title', 'No Title')
#                 link = res.get('href', 'No Link')
#                 # 截取前 200 个字符避免 Token 爆炸
#                 body = res.get('body', '')[:200].replace("\n", " ") 
                
#                 output.append(f"\n[{i}] {title}")
#                 output.append(f"    Link: {link}")
#                 output.append(f"    Snippet: {body}...")
            
#             return "\n".join(output)

#         except Exception as e:
#             return f"Search execution failed: {str(e)}"





# src/minisweagent/tools/web_search.py
import http.client
import json

# ==============================
# 请在这里填入你的 API Key
API_KEY = "e34113db4af9cb7c10abc59dd3c5d4e81a947214"
# ==============================

class SearchTool:
    def __init__(self):
        if not API_KEY or "你的" in API_KEY:
            print("Warning: Serper API Key is not set in tools/web_search.py")

    def run(self, query: str, max_results: int = 5) -> str:
        if not API_KEY:
            return "Error: Serper API Key is missing. Please edit `src/minisweagent/tools/web_search.py`."

        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({"q": query, "num": max_results})
            headers = {'X-API-KEY': API_KEY, 'Content-Type': 'application/json'}
            
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            
            results = json.loads(data)
            
            if 'organic' not in results:
                return f"No organic results found for: {query}"

            output = [f"Google Search Results for '{query}':"]
            for i, item in enumerate(results['organic'], 1):
                title = item.get('title', 'No Title')
                link = item.get('link', 'No Link')
                snippet = item.get('snippet', '')[:200].replace("\n", " ")

                output.append(f"\n[{i}] {title}")
                output.append(f"    Link: {link}")
                output.append(f"    Snippet: {snippet}...")
            
            return "\n".join(output)

        except Exception as e:
            return f"Search execution failed: {str(e)}"

