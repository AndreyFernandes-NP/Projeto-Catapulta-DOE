import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import get_csv, load_csv

if __name__ == "__main__":
    fontes = get_csv()
    print(f"Fontes encontradas: {[f['nome'] for f in fontes]}")

    for fonte in fontes:
        try:
            df = load_csv(fonte)
            print(f"'{fonte['nome']}' carregado com {len(df)} linhas e {len(df.columns)} colunas.")
        except Exception as e:
            print(f"Erro ao carregar '{fonte['nome']}': {e}")