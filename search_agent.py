from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_tavily import TavilySearch

from typing import List
from pydantic import BaseModel, Field

class source(BaseModel):
    """Schema for a source used by agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources"""
    answer: str = Field(description="the agent answer to the query")
    sources: List[source] = Field(default_factory=list, description="A list of sources used to generate the answer")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)   

def main():
    result = agent.invoke(
            {
                "messages":[HumanMessage(content="search for job opening for AI engineer on linkedIn in Noida Area")]
            }
    )
    print(result)
    

if __name__ == "__main__":
    main()
