# import os
# from dotenv import load_dotenv
# import fitz  # PyMuPDF for PDF processing
# from agno.embedder.google import GeminiEmbedder
# from pinecone import Pinecone

# # Load environment variables
# load_dotenv()

# # Load credentials
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# PINECONE_REGION = os.getenv("PINECONE_REGION") or "us-east-1"

# # Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Check if index exists
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     raise ValueError(f"Index '{PINECONE_INDEX_NAME}' does not exist. Please create it first.")

# # Connect to the index
# index = pc.Index(PINECONE_INDEX_NAME)

# # Initialize Gemini embedder
# embedder = GeminiEmbedder(api_key=GEMINI_API_KEY, dimensions=512)

# # Directory with PDF files
# documents_dir = "data_ingestion/data"

# def load_documents(directory):
#     documents = []
#     for filename in os.listdir(directory):
#         if filename.lower().endswith(".pdf"):
#             file_path = os.path.join(directory, filename)
#             try:
#                 with fitz.open(file_path) as doc:
#                     text = ""
#                     for page in doc:
#                         text += page.get_text()
#                     documents.append({
#                         "id": filename,
#                         "content": text
#                     })
#             except Exception as e:
#                 print(f"❌ Error processing {filename}: {e}")
#     return documents

# # Load and embed documents
# docs = load_documents(documents_dir)
# print(f"📄 Loaded {len(docs)} documents.")

# for doc in docs:
#     embedding = embedder.get_embedding(doc["content"])
#     if len(embedding) != 512:
#         raise ValueError(f"Embedding dimension mismatch for {doc['id']}! Got {len(embedding)} instead of 512.")

#     index.upsert(vectors=[
#         (doc["id"], embedding, {"filename": doc["id"]})
#     ])
#     print(f"✅ Uploaded {doc['id']} to Pinecone.")

# print("🚀 All documents uploaded successfully.")



import os
import time
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF processing
from agno.embedder.google import GeminiEmbedder
from pinecone import Pinecone
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.genai.errors import ClientError

# Load environment variables
load_dotenv()

# Retrieve credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{PINECONE_INDEX_NAME}' does not exist. Please create it.")

index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Gemini embedder (ensure dimensions match your Pinecone index)
embedder = GeminiEmbedder(api_key=GEMINI_API_KEY, dimensions=512)

# Retry decorator for Gemini embedding calls
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(ClientError)
)
def safe_embed(text):
    return embedder.get_embedding(text)

# PDF directory
documents_dir = "data_ingestion/data"

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            try:
                with fitz.open(file_path) as doc:
                    text = "\n".join([page.get_text() for page in doc])
                    documents.append({
                        "id": filename,
                        "content": text
                    })
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
    return documents

# Load and embed
docs = load_documents(documents_dir)
print(f"📄 Loaded {len(docs)} documents.")

for doc in docs:
    try:
        embedding = safe_embed(doc["content"])
        if len(embedding) != 512:
            raise ValueError(f"Embedding dimension mismatch: got {len(embedding)} instead of 512.")
        index.upsert([(doc["id"], embedding, {"filename": doc["id"]})])
        print(f"✅ Uploaded {doc['id']} to Pinecone.")
        time.sleep(1)  # Respect rate limits
    except Exception as e:
        print(f"❌ Failed to embed/upload {doc['id']}: {e}")

print("🚀 All documents processed.")
