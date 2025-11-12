from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time
from rag import SimpleRAG
import config

app = FastAPI(title="Assistente Virtual - TEC II (RAG + Ollama)")

rag = SimpleRAG("sentence-transformers/all-MiniLM-L6-v2", 500, 50)
rag.load_document("data/base.txt") 
rag.create_index()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest):
    start = time.time()
    retrieved = rag.retrieve(request.question, 1)
    context = "\n\n".join(retrieved)
    prompt = f"""
    Você é um assistente que deve responder apenas com base no conteúdo abaixo.
    Se a resposta não estiver presente no texto, diga: "Não encontrei essa informação no documento."

    Contexto:
    {context}

    Pergunta: {request.question}
    """
    response = requests.post(config.OLLAMA_URL, json={
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Responda APENAS com base no contexto fornecido."},
            {"role": "user", "content": prompt}
    ]
})


    if response.status_code == 200:
        data = response.json()
        answer = data.get("message", {}).get("content", "Não consegui gerar uma resposta.")
    else:
        answer = f"Erro ao conectar com o Ollama: {response.status_code}"

    latency = round((time.time() - start) * 1000, 2)
    return {
        "answer": answer.strip(),
        "retrieved_chunks": retrieved,
        "latency_ms": latency
    }