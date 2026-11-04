from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv

# load_dotenv()

# os.getenv("GROQ_API_KEY")
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatGroq(model="qwen/qwen3.8-27b")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm