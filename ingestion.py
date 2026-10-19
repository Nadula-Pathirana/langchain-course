import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# pyrefly: ignore[missing-import]
from langchain_pinecone import PineconeVectorStore
# pyrefly: ignore[missing-import]
from langchain_text_splitters import CharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_unstructured import UnstructuredLoader
from openai.types import embedding

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredLoader(file_path="/Users/nadulapathirana/Desktop/langchain-course/mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    #print(document)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)

    print(len(chunks))

    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print("ingesting...")

    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("finished")





