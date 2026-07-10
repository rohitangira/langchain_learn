import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Starting ingestion process...")
    loader = TextLoader("/media/rohit/8c7a33c8-5d37-48e3-9e08-180e261b8665/home/rohit/workspace/langchain_learn/mediumblog.txt")
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Number of chunks created: {len(texts)}")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print("Creating Pinecone vector store...")
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ.get("INDEX_NAME"),
    )
    pass

