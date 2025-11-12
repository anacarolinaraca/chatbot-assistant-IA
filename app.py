from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import requests
import time
from rag import SimpleRAG
import config

app = FastAPI(title="Assistente Virtual - TEC II (RAG + Ollama)")

rag = None


@app.on_event("startup")
def startup_event():
    global rag
    print("Inicializando RAG (pode demorar alguns minutos)...")
    rag = SimpleRAG(config.MODEL_NAME, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    rag.load_document("data/base.md")
    rag.create_index()
    print("RAG inicializado com sucesso.")

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        start = time.time()
        retrieved = rag.retrieve(request.question, 1)
        context = "\n\n".join(retrieved)
        prompt = f"""
        Você é um assistente que deve responder apenas com base no conteúdo abaixo.
        Se a resposta não estiver presente no texto, diga: "Não encontrei essa informação no documento."\n
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
        }, stream=False)

        if response.status_code == 200:
            answer = None
            try:
                data = response.json()
                answer = data.get("message", {}).get("content")
            except Exception:
                parts = []
                for line in response.text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    msg = obj.get("message", {}).get("content")
                    if msg:
                        parts.append(msg)
                if parts:
                    answer = "".join(parts)

            if not answer:
                return JSONResponse(status_code=502, content={
                    "answer": "Erro ao parsear resposta do Ollama.",
                    "ollama_status": response.status_code,
                    "ollama_text": response.text
                })
        else:
            return JSONResponse(status_code=502, content={
                "answer": "Erro ao conectar com o Ollama.",
                "ollama_status": response.status_code,
                "ollama_text": response.text
            })

        latency = round((time.time() - start) * 1000, 2)
        return {
            "answer": answer.strip(),
            "retrieved_chunks": retrieved,
            "latency_ms": latency
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse(status_code=500, content={
            "answer": "Erro interno no servidor.",
            "error": str(e),
            "traceback": tb
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)