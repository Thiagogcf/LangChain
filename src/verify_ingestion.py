import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

def verify_ingestion():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")

    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_name = os.getenv("POSTGRES_DB")
    connection_string = f"postgresql://{db_user}:{db_password}@localhost:5433/{db_name}"
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    
    # Conectar ao vector store existente
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="my_collection",
        connection=connection_string,
    )
    
    # Fazer uma busca de teste para verificar se há documentos
    test_query = "test"
    results = vectorstore.similarity_search(test_query, k=5)
    
    print(f"Número de documentos encontrados: {len(results)}")
    
    if results:
        print("\n=== DOCUMENTOS ENCONTRADOS ===")
        for i, doc in enumerate(results, 1):
            print(f"\nDocumento {i}:")
            print(f"Conteúdo (primeiros 200 chars): {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
    else:
        print("Nenhum documento foi encontrado no banco de dados.")
        print("Verifique se a ingestão foi executada corretamente.")

if __name__ == "__main__":
    verify_ingestion()