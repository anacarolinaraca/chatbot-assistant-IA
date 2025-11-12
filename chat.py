import requests

url = "http://127.0.0.1:8000/chat"

while True:
    question = input("Pergunta: ")
    if question.lower() in ["sair", "exit"]:
        break

    response = requests.post(url, json={"question": question})
    data = response.json()
    
    print("\nResposta:")
    print(data["answer"])
    print("-"*40)