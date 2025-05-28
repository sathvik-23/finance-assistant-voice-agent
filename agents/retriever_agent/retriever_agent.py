# agents/retriever_agent/retriever_agent.py

import os
from dotenv import load_dotenv

from pinecone import Pinecone                              # Pinecone v7.x SDK
from agno.embedder.google import GeminiEmbedder            # Gemini embeddings
from google import genai                                   # Google GenAI SDK (google-genai)

# 1️⃣ Load environment variables
load_dotenv()
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_ENV        = os.getenv("PINECONE_ENVIRONMENT")      # your Pinecone environment
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")       # e.g. "qa-bot"
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")

# 2️⃣ Init Pinecone client (no top-level init())
pc    = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(PINECONE_INDEX_NAME)

# 3️⃣ Instantiate the GenAI client with your API key (no configure call)
client = genai.Client(api_key=GEMINI_API_KEY)

# 4️⃣ Initialize Gemini embedder for 512-dimensional embeddings
embedder = GeminiEmbedder(api_key=GEMINI_API_KEY, dimensions=512)

def retrieve_and_answer(question: str, top_k: int = 5, snippet_chars: int = 300):
    # a) Embed the question
    q_emb = embedder.get_embedding(question)

    # b) Query Pinecone for top-k matches
    resp = index.query(vector=q_emb, top_k=top_k, include_metadata=True)

    # c) Build context snippets
    snippets = []
    for m in resp.matches:
        fn  = m.metadata.get("filename", "<unknown>")
        txt = m.metadata.get("text", "")[:snippet_chars]
        snippets.append(f"**{fn}**:\n{txt}")

    context = "\n\n---\n\n".join(snippets)

    # d) Construct a single-prompt payload
    prompt = (
        "You are a financial analyst. Using ONLY the snippets below, answer the question precisely.\n\n"
        f"{context}\n\n"
        f"**Question:** {question}\n**Answer:**"
    )

    # e) Generate the answer via the GenAI SDK
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    print(response.text)

if __name__ == "__main__":
    retrieve_and_answer("What adobe's data ?")
