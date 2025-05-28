import os
import time
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF processing
from agno.embedder.google import GeminiEmbedder
from pinecone import Pinecone
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.genai.errors import ClientError

# 1️⃣ Load environment variables
load_dotenv()
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_ENV        = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 2️⃣ Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{PINECONE_INDEX_NAME}' does not exist.")
index = pc.Index(PINECONE_INDEX_NAME)

# 3️⃣ Clear all existing vectors
index.delete(delete_all=True)
print(f"🗑️ Cleared all data from index '{PINECONE_INDEX_NAME}'")

# 4️⃣ Initialize Gemini embedder
embedder = GeminiEmbedder(api_key=GEMINI_API_KEY, dimensions=512)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(ClientError)
)
def safe_embed(text: str):
    return embedder.get_embedding(text)

# 5️⃣ Simple chunker (character‐based)
def chunk_text(text: str, max_chars: int = 1000):
    return [text[i : i + max_chars].strip()
            for i in range(0, len(text), max_chars)
            if text[i : i + max_chars].strip()]

# 6️⃣ Path to your Financial_Summary.pdf
pdf_path = os.path.join(os.path.dirname(__file__), "data", "Financial_Summary.pdf")
if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"{pdf_path} not found")

# 7️⃣ Open, read, chunk, embed & upsert
doc = fitz.open(pdf_path)
full_text = "\n".join(page.get_text() for page in doc)
chunks = chunk_text(full_text)
print(f"📄 Financial_Summary.pdf: splitting into {len(chunks)} chunks")

for i, chunk in enumerate(chunks):
    try:
        emb = safe_embed(chunk)
        metadata = {"filename": "Financial_Summary.pdf", "text": chunk}
        vector_id = f"Financial_Summary-{i}"
        index.upsert([(vector_id, emb, metadata)])
        print(f"  ✅ upserted chunk {i}")
        time.sleep(0.2)  # throttle to respect rate limits
    except Exception as e:
        print(f"  ❌ failed chunk {i}: {e}")

print("🚀 All chunks of Financial_Summary.pdf uploaded successfully.")
