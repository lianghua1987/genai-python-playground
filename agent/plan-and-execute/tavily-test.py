from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
client = TavilyClient("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")

response = client.search(
    query="research long-term economic, technological, and social trends expected by 2026"
)
print(response)




