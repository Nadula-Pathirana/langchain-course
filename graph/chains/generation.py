from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq
import os 
# pyrefly: ignore[unknown-name]
llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=os.getenv("GROQ_API_KEY"), temperature=0)

# Inline equivalent of hub.pull("rlm/rag-prompt") — the hub module was removed
# from the langchain package, and pulling public prompts now requires a
# LangSmith API key.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)

generation_chain = prompt | llm | StrOutputParser()