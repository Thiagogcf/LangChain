# Ingestão e Busca Semântica com LangChain e Postgres

Este projeto permite a ingestão de um documento PDF em um banco de dados PostgreSQL com pgVector e a busca semântica de informações no documento através de um CLI.

## Tecnologias

- Python
- LangChain
- PostgreSQL + pgVector
- Docker & Docker Compose
- Google Gemini

## Estrutura do Projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .env
├── src/
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
├── document.pdf
└── README.md
```

## Como Executar

1. **Clone o repositório:**

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   ```

2. **Crie e ative um ambiente virtual:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as variáveis de ambiente:**

   - Renomeie o arquivo `.env.example` para `.env`.
   - Adicione sua `GOOGLE_API_KEY` no arquivo `.env`.

5. **Suba o banco de dados:**

   ```bash
   docker compose up -d
   ```

6. **Adicione o arquivo `document.pdf`:**

   - Adicione o arquivo `document.pdf` que você deseja ingerir na raiz do projeto.

7. **Execute a ingestão do PDF:**

   ```bash
   python src/ingest.py
   ```

8. **Inicie o chat:**

   ```bash
   python src/chat.py
   ```

9. **Interaja com o chat:**

   - Faça suas perguntas no terminal.
   - Para sair, digite `sair`.
