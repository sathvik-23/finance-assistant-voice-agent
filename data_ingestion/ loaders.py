import os
from dotenv import load_dotenv
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

# Define index name
index_name = os.getenv("PINECONE_INDEX_NAME")

# Create Pinecone index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

# Load documents from the 'data' directory
loader = DirectoryLoader('./data', glob="**/*.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Generate embeddings
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Upload to Pinecone
vectorstore = LangchainPinecone.from_documents(docs, embedding, index_name=index_name)

print(f"Successfully uploaded {len(docs)} documents to Pinecone index '{index_name}'.")
