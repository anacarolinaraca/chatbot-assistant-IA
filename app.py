from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time
from rag import SimpleRAG
import config

app = FastAPI(title="Assistente Virtual - TEC II (RAG + Ollama)")

# Inicializamos o RAG no evento de startup para deixar os logs visíveis
# e informar o usuário sobre o progresso (essa etapa pode demorar).
rag = None


@app.on_event("startup")
def startup_event():
    global rag
    print("Inicializando RAG (pode demorar alguns minutos)...")
    # Usar as configurações definidas em config.py
    rag = SimpleRAG(config.MODEL_NAME, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    rag.load_document("data/base.md")
    rag.create_index()
    print("RAG inicializado com sucesso.")

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


if __name__ == "__main__":
    # Ao executar `python app.py` queremos iniciar o servidor Uvicorn
    # Usamos o módulo path "app:app" para habilitar reload durante desenvolvimento
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)