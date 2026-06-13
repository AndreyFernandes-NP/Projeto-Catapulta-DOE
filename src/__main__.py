import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import get_csv, load_csv
from src.supervised import Catapulta

if __name__ == "__main__":
    print(f"[Execução] Programa iniciado, configurando pipeline...")

    fontes = get_csv()
    cat_cls = Catapulta()

    if not fontes:
        raise FileNotFoundError("[Erro] Nenhum arquivo CSV encontrado em 'data/'")
    
    print(f"[Execução] Pipeline configurada, {len(fontes)} arquivo(s) CSV encontrado(s) em 'data/'.")

    for fonte in fontes:
        try:
            print(f"[Execução] Carregando dataset '{fonte["nome"]}'...")

            df = load_csv(fonte)
            cat_cls.store_dataset(fonte["nome"], df)
            print(f"[Execução] '{fonte['nome']}' carregado com {len(df)} linhas e {len(df.columns)} colunas.")

            print(f"[Execução] Executando ajuste de curva...")
            df_adj = cat_cls.curve_adjustment(df, 2)
            cat_cls.store_polynomial(fonte["nome"], df_adj)
            print(f"[Execução] Ajuste executado, polinômio de '{fonte["nome"]}' criado e salvo.")
        except Exception as e:
            print(f"[Erro] Ocorreu uma falha durante o processamento do dataset '{fonte['nome']}': {e}")
    
    print(f"Fontes disponíveis: {cat_cls.get_fonts()}")
    print(cat_cls.parameters) # cat_cls.get_parameters() também funciona caso queira pegar de um tipo específico além do 'first' padrão
    cat_cls.analyze_datasets()
    