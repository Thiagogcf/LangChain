import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

def validate_environment():
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
        sys.exit(1)

def validate_pdf_file():
    pdf_paths = ["document.pdf", "../document.pdf"]
    
    for pdf_file in pdf_paths:
        if os.path.exists(pdf_file):
            print(f"✅ Arquivo PDF encontrado: {pdf_file}")
            return pdf_file
    
    print("❌ Erro: Arquivo 'document.pdf' não encontrado.")
    print("Certifique-se de que o arquivo PDF está na raiz do projeto ou na pasta atual.")
    sys.exit(1)

def ingest_pdf():
    try:
        print("🚀 Iniciando processo de ingestão...")
        
        validate_environment()
        pdf_file = validate_pdf_file()
        
        print("📖 Carregando documento PDF...")
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        
        if not documents:
            print("❌ Erro: Nenhum conteúdo foi extraído do PDF.")
            sys.exit(1)
        
        print(f"✅ PDF carregado com sucesso. Total de páginas: {len(documents)}")
        
        print("✂️  Dividindo documento em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        if not docs:
            print("❌ Erro: Nenhum chunk foi criado do documento.")
            sys.exit(1)
        
        print(f"✅ Documento dividido em {len(docs)} chunks de texto.")
        
        print("🔄 Configurando embeddings do Google Gemini...")
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
        
        print("💾 Conectando ao banco de dados PostgreSQL...")
        print("⏳ Gerando embeddings e salvando no banco (isso pode levar alguns minutos)...")
        
        db = PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=collection_name,
            connection=connection_string,
        )
        
        print("🎉 Ingestão concluída com sucesso!")
        print(f"📊 Total de chunks processados: {len(docs)}")
        print("🔍 Agora você pode usar o chat.py para fazer perguntas sobre o documento.")
        
    except FileNotFoundError as e:
        print(f"❌ Erro: Arquivo não encontrado - {e}")
        sys.exit(1)
    except ConnectionError as e:
        print(f"❌ Erro de conexão: {e}")
        print("Verifique se o PostgreSQL está rodando com: docker compose up -d")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro durante a ingestão: {e}")
        print("Verifique suas configurações e tente novamente.")
        sys.exit(1)

if __name__ == "__main__":
    ingest_pdf()
