import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

def validate_search_environment():
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

def similarity_search(query):
    try:
        if not query or not query.strip():
            print("❌ Erro: Pergunta não pode estar vazia.")
            return []
        
        if not validate_search_environment():
            return []
        
        google_api_key = os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=google_api_key
        )
        
        db_user = os.getenv("POSTGRES_USER")
        db_password = os.getenv("POSTGRES_PASSWORD")
        db_name = os.getenv("POSTGRES_DB")
        connection_string = f"postgresql://{db_user}:{db_password}@localhost:5433/{db_name}"
        collection_name = "my_collection"
        
        db = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_string,
        )
        
        results = db.similarity_search_with_score(query, k=10)
        
        if not results:
            print("⚠️  Nenhum resultado encontrado para a busca.")
            return []
        
        return results
        
    except ConnectionError as e:
        print(f"❌ Erro de conexão com o banco de dados: {e}")
        print("Verifique se o PostgreSQL está rodando com: docker compose up -d")
        return []
    
    except Exception as e:
        print(f"❌ Erro durante a busca: {e}")
        print("Certifique-se de que a ingestão foi executada com sucesso.")
        return []
