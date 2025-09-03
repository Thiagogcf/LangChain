import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

def similarity_search(query):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

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

    return results
