import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from search import similarity_search

load_dotenv()

def validate_chat_environment():
    required_vars = ["GOOGLE_API_KEY", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Erro: As seguintes variáveis de ambiente não foram encontradas no arquivo .env:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nVerifique o arquivo .env e adicione as variáveis necessárias.")
        return False

    return True

def print_welcome():
    print("=" * 60)
    print("🤖 CHAT COM SEU DOCUMENTO PDF")
    print("=" * 60)
    print("💡 Faça perguntas sobre o conteúdo do seu documento.")
    print("💡 Digite 'sair' para encerrar o chat.")
    print("=" * 60)

def format_response(response_content):
    return f"\n💬 RESPOSTA: {response_content}\n" + "-" * 60

def main():
    try:
        if not validate_chat_environment():
            sys.exit(1)

        print("🔄 Inicializando chat...")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        try:
            llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite",
                google_api_key=google_api_key
            )
        except Exception as e:
            print(f"❌ Erro ao configurar o modelo: {e}")
            print("Verifique sua GOOGLE_API_KEY e conexão com a internet.")
            sys.exit(1)

        print_welcome()

        while True:
            try:
                user_question = input("\n❓ Faça sua pergunta: ").strip()

                if user_question.lower() in ['sair', 'exit', 'quit']:
                    print("\n👋 Encerrando chat. Até logo!")
                    break

                if not user_question:
                    print("⚠️  Por favor, digite uma pergunta válida.")
                    continue

                print("🔍 Buscando informações relevantes...")

                search_results = similarity_search(user_question)

                if not search_results:
                    print("❌ Não foi possível encontrar informações relevantes.")
                    print("Certifique-se de que a ingestão foi executada com sucesso.")
                    continue

                context = "\n".join([doc.page_content for doc, score in search_results])

                if not context.strip():
                    print("⚠️  Contexto vazio encontrado. Tentando novamente...")
                    continue

                prompt = f"""CONTEXTO:
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

RESPONDA A "PERGUNTA DO USUÁRIO" """

                print("🤔 Processando sua pergunta...")

                response = llm.invoke(prompt)

                print(format_response(response.content))

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrompido. Até logo!")
                break

            except Exception as e:
                print(f"\n❌ Erro ao processar pergunta: {e}")
                print("Tente novamente ou digite 'sair' para encerrar.")
                continue

    except Exception as e:
        print(f"❌ Erro fatal no chat: {e}")
        print("Verifique suas configurações e tente novamente.")
        sys.exit(1)

if __name__ == "__main__":
    main()
