from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_tavily import TavilySearch


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)   

def main():
    result = agent.invoke({"messages":[HumanMessage(content="search for job opening for AI engineer on linkedIn in Noida Area")]})
    print(result)
    

if __name__ == "__main__":
    main()
