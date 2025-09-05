import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

def ingest_pdf():
    print("Iniciando ingestão do PDF...")
    
    print(f"Carregando PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    print("Dividindo documento em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documento dividido em {len(chunks)} chunks")
    
    use_gemini = GOOGLE_API_KEY and GOOGLE_API_KEY.strip()
    use_openai = OPENAI_API_KEY and OPENAI_API_KEY.strip()
    
    if not use_gemini and not use_openai:
        print("Erro: Nenhuma API key configurada (Google ou OpenAI)")
        return
        
    if use_gemini:
        try:
            print("Configurando embeddings Google Gemini...")
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
            else:
                raise e
    else:
        print("Configurando embeddings OpenAI...")
        embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
    
    if use_gemini:
        collection_name = f"{PG_VECTOR_COLLECTION_NAME}_gemini"
        print("Usando coleção para embeddings Gemini...")
    else:
        collection_name = f"{PG_VECTOR_COLLECTION_NAME}_openai"  
        print("Usando coleção para embeddings OpenAI...")
    
    print("Conectando ao banco de dados...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    
    print("Armazenando embeddings no banco de dados...")
    vector_store.add_documents(chunks)
    
    print("Ingestão concluída com sucesso!")


if __name__ == "__main__":
    ingest_pdf()