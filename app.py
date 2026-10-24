from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from ollama import chat
import chromadb

app = FastAPI()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load persistent ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="ppe_docs"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question

    # 1. Convert question into embedding
    question_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True
    )

    # 2. Retrieve top chunks
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=5
    )

    # 3. Filter weak matches
    filtered_docs = []
    filtered_meta = []

    distance_threshold = 0.7

    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if distance <= distance_threshold:
            filtered_docs.append(text)
            filtered_meta.append(meta)

    # 4. No relevant information
    if not filtered_docs:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": []
        }

    # 5. Build context
    context = ""

    for i, (text, meta) in enumerate(
        zip(filtered_docs, filtered_meta),
        start=1
    ):
        context += f"""
[Source {i}]
File: {meta['source']}
Page: {meta['page']}

{text}

"""

    # 6. Build prompt
    prompt = f"""
You are answering questions using retrieved document context.

Use only the context below.

If the answer is not present in the context, say:
"I could not find this information in the provided documents."

When answering, mention the source number where appropriate.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    # 7. Send to Ollama
    response = chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.message.content

    # 8. Prepare sources
    sources = []

    for meta in filtered_meta:
        source_item = {
            "file": meta["source"],
            "page": meta["page"]
        }

        if source_item not in sources:
            sources.append(source_item)

    # 9. Return API response
    return {
        "answer": answer,
        "sources": sources
    }