# Sistema de Ingestão e Busca Semântica com LangChain

Sistema desenvolvido para ingestão de documentos PDF e busca semântica usando LangChain, PostgreSQL com pgVector e Gemini.

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8+
- Docker e Docker Compose
- API Key do Google Gemini configurada no arquivo `.env`

### Passo 1: Configurar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### Passo 2: Instalar Dependências

```bash
pip install -r requirements.txt
```

### Passo 3: Subir o Banco de Dados

```bash
docker compose up -d
```

Aguarde a inicialização do PostgreSQL e crie a extensão vector:

```bash
docker exec langchain-postgres-1 psql -U user -d vectordb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Este comando irá:
- Subir PostgreSQL com extensão pgVector na porta 5433
- Criar a extensão vector manualmente

### Passo 4: Executar Ingestão do PDF

```bash
python src/ingest.py
```

Este script irá:
- Carregar o arquivo `document.pdf`
- Dividir o texto em chunks de 1000 caracteres com overlap de 150
- Gerar embeddings usando Gemini `models/embedding-001`
- Armazenar no PostgreSQL com pgVector

### Passo 5: Rodar o Chat

```bash
python src/chat.py
```

## 📝 Como Usar

O sistema responderá apenas com informações contidas no PDF ingerido. Exemplos:

```
Faça sua pergunta:

PERGUNTA: Qual o faturamento da empresa?
RESPOSTA: [Resposta baseada no conteúdo do PDF]

---

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Para sair do chat, digite: `sair`, `quit`, `exit` ou `q`.

## ⚙️ Configuração

O arquivo `.env` contém as configurações necessárias:

- `GOOGLE_API_KEY`: Chave da API do Google
- `DATABASE_URL`: URL de conexão com PostgreSQL
- `PDF_PATH`: Caminho para o arquivo PDF
- `PG_VECTOR_COLLECTION_NAME`: Nome da coleção no banco

## 🛠️ Tecnologias Utilizadas

- **LangChain**: Framework para aplicações com LLM
- **PostgreSQL + pgVector**: Banco vetorial
- **Google Gemini**: Embeddings (`models/embedding-001`) e LLM (`gemini-2.5-flash-lite`)
- **Docker**: Containerização do banco de dados