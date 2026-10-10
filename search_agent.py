from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
# pyrefly: ignore [missing-import]
from langchain_tavily import TavilySearch

tavily = TavilySearch(max_results=3)
tools = [tavily]

# @tool
# def search(query: str) -> str:
#     """
#     Search the internet for information, including job informations.
#     Use this tool whenever the user asks for current information.
#     """
#     print(f"Searching for {query}")
#     response = tavily.search(query=query)
#     return response['results'][0]['content']

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="URL of the source ")

class AgentResponse(BaseModel):
    """Schema for the agent response with answers and sources."""
    answers:str = Field(description="agent's answers to the  query")
    sources:List[Source] = Field(default_factory=list ,description="list of sources used to genarate the answer")


    

llm = init_chat_model("openrouter:apodex/apodex-1.1-mini:free", temperature=0)

agent = create_agent(model=llm,tools=tools,response_format=AgentResponse)


def main():
    print("hello")
    response = agent.invoke({
        "messages": [
            HumanMessage(content="search for 3 AI Engineer jobs in Colombo")
        ]
    })
    print(response.get("structured_response") or response["messages"][-1].content)


if __name__ == "__main__":
    main()
    

