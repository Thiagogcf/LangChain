import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from search import similarity_search

load_dotenv()

def main():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", google_api_key=google_api_key)

    while True:
        user_question = input("Faça sua pergunta: ")

        if user_question.lower() == "sair":
            break

        search_results = similarity_search(user_question)

        context = "\n".join([doc.page_content for doc, score in search_results])

        prompt = f"""
        CONTEXTO:
        {context}

        REGRAS:
        - Responda somente com base no CONTEXTO.
        - Se a informação não estiver explicitamente no CONTEXTO, responda:
          "Não tenho informações necessárias para responder sua pergunta."
        - Nunca invente ou use conhecimento externo.
        - Nunca produza opiniões ou interpretações além do que está escrito.

        EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
        Pergunta: "Qual é a capital da França?"
        Resposta: "Não tenho informações necessárias para responder sua pergunta."

        Pergunta: "Quantos clientes temos em 2024?"
        Resposta: "Não tenho informações necessárias para responder sua pergunta."

        Pergunta: "Você acha isso bom ou ruim?"
        Resposta: "Não tenho informações necessárias para responder sua pergunta."

        PERGUNTA DO USUÁRIO:
        {user_question}

        RESPONDA A "PERGUNTA DO USUÁRIO"
        """

        response = llm.invoke(prompt)

        print(f"RESPOSTA: {response.content}")

if __name__ == "__main__":
    main()
