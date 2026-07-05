from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from tavily import TavilyClient


tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    tool that searches the web for a given query and returns the result
    Arguments:
    query: str - the query to search for
    Returns:
    str - the result of the search
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
tools = [search]
agent = create_agent(model=llm, tools=tools)   

def main():
    result = agent.invoke({"messages":[HumanMessage(content="search for job opening for AI engineer on linkedIn in Noida Area")]})
    print(result)
    

if __name__ == "__main__":
    main()
