import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain.schema import HumanMessage

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

GOOGLE_LLM_MODEL = "gemini-2.5-flash-lite"
OPENAI_LLM_MODEL = "gpt-5-nano"

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

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
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question=None):
    if not question:
        return None
        
    try:
        use_gemini = GOOGLE_API_KEY and GOOGLE_API_KEY.strip()
        use_openai = OPENAI_API_KEY and OPENAI_API_KEY.strip()
        
        if not use_gemini and not use_openai:
            print("Erro: Nenhuma API key configurada (Google ou OpenAI)")
            return None
            
        if use_gemini:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=GOOGLE_EMBEDDING_MODEL,
                    google_api_key=GOOGLE_API_KEY
                )
            except Exception as e:
                if use_openai:
                    print(f"Erro com Gemini embeddings, usando OpenAI: {e}")
                    embeddings = OpenAIEmbeddings(
                        model=OPENAI_EMBEDDING_MODEL,
                        openai_api_key=OPENAI_API_KEY
                    )
                    use_gemini = False
                else:
                    raise e
        else:
            embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
        
        if use_gemini:
            collection_name = f"{PG_VECTOR_COLLECTION_NAME}_gemini"
        else:
            collection_name = f"{PG_VECTOR_COLLECTION_NAME}_openai"
        
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=DATABASE_URL,
            use_jsonb=True,
        )
        
        docs_with_scores = vector_store.similarity_search_with_score(question, k=10)
        
        contexto = "\n\n".join([doc.page_content for doc, score in docs_with_scores])
        
        prompt_completo = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
        
        if use_gemini:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=GOOGLE_LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0
                )
            except Exception as e:
                if use_openai:
                    print(f"Erro com Gemini LLM, usando OpenAI: {e}")
                    llm = ChatOpenAI(
                        model=OPENAI_LLM_MODEL,
                        openai_api_key=OPENAI_API_KEY,
                        temperature=0
                    )
                else:
                    raise e
        else:
            llm = ChatOpenAI(
                model=OPENAI_LLM_MODEL,
                openai_api_key=OPENAI_API_KEY,
                temperature=0
            )
        
        response = llm.invoke([HumanMessage(content=prompt_completo)])
        
        return response.content
        
    except Exception as e:
        print(f"Erro ao processar pergunta: {e}")
        return None