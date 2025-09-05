from search import search_prompt

def main():
    print("=== Sistema de Busca Semântica ===")
    print("Digite 'sair' ou 'quit' para encerrar o chat.")
    print()
    
    while True:
        try:
            pergunta = input("Faça sua pergunta:\n\nPERGUNTA: ").strip()
            
            if pergunta.lower() in ['sair', 'quit', 'exit', 'q']:
                print("Encerrando chat...")
                break
            
            if not pergunta:
                print("Por favor, digite uma pergunta válida.")
                continue
            
            print("Processando...")
            resposta = search_prompt(pergunta)
            
            if resposta:
                print(f"RESPOSTA: {resposta}")
            else:
                print("RESPOSTA: Erro ao processar sua pergunta. Tente novamente.")
            
            print("\n" + "="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nEncerrando chat...")
            break
        except Exception as e:
            print(f"Erro inesperado: {e}")
            break

if __name__ == "__main__":
    main()