from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
# pyrefly: ignore [missing-import]
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Search the internet for information, including job informations.
    Use this tool whenever the user asks for current information.
    """
    print(f"Searching for {query}")
    response = tavily.search(query=query)
    return response['results'][0]['content']


llm = init_chat_model("openrouter:google/gemini-2.5-flash-lite",temperature=0)
tools = [search]
agent = create_agent(model=llm,tools=tools)


def main():
    print("hello")
    response = agent.invoke({"messages":HumanMessage(content="search for 3 AI Enginner jobs in colombo")})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
    

