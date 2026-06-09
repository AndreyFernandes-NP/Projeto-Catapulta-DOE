import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import get_csv, load_csv

if __name__ == "__main__":
    fontes = get_csv()

    if not fontes:
        raise FileNotFoundError("[Erro] Nenhum arquivo CSV encontrado em 'data/'")

    for fonte in fontes:
        try:
            df = load_csv(fonte)
            print(f"'{fonte['nome']}' carregado com {len(df)} linhas e {len(df.columns)} colunas.") # Remover depois
            # TODO: Exploração inicial dos dados, como tipos de colunas, estatísticas básicas, .head(), e etc.
            # TODO: Fazer ajuste de curvas e resolver otimização
        except Exception as e:
            print(f"Erro ao carregar '{fonte['nome']}': {e}")