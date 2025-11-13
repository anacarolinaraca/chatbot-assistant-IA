# Projeto Assitente de Chatbot IA

## Sobre

Este projeto implementa um sistema RAG (Retrieval-Augmented Generation) que permite ao assistente virtual responder perguntas usando uma base de conhecimento relacionadas à saúde mental.

## Tecnologias Utilizadas

- Python
- Ollama
- Sentence Transformers
- FAISS
- FastAPI
- LLM: gemma:2b

## Como Usar 

1. Clone o repositório: 
```
git clone git@github.com:anacarolinaraca/chatbot-assistant-IA.git
```

2. Acesse o diretório do projeto:
```
cd /chatbot-assistant-IA.git
```

3. Crie e ative o ambiente virtual:
```
python -m venv venv
venv\Scripts\activate
```

4. Instale as dependências:
```
pip install -r requirements.txt
```

5. Configure o Ollama:
Baixe o Ollama [aqui](https://ollama.com/download/windows).
Após a instalação, baixe o modelo:
```
ollama pull gemma:2b
```
Inicie o servidor do Ollama:
```
ollama serve
```

6. Inicie o servidor FastAPI
```
python app.py
```

7. Rode o chat para fazer perguntas
```
python chat.py
```
