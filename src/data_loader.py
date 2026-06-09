"""
data_loader.py
--------------
Responsável por carregar os dados brutos além de fornecer funções auxiliares para manipulação de DataFrames e pré-processamento básico.

Os dados carregados devem vir diretamente de data/catapulta_doe.csv.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.__config__ import PATHS

DATA_DIR = PATHS.data

encodings = ["utf-8", "utf-8-sig", "latin1", "iso-8859-1", "cp1252"]
na_values = ['nan', '?', 'null', '0', '-']
separators = [";", ","]

# Utils

def has_replacement_char(df: pd.DataFrame) -> bool:
    return any("�" in str(col) for col in df.columns)

def get_csv() -> list[dict]:
    caminho = DATA_DIR
    files = []

    for arquivo in caminho.glob("*.csv"):
        files.append({"nome": arquivo.stem, "arquivo": arquivo.name})

    return files

def load_csv(fonte: dict) -> pd.DataFrame:
    caminho = DATA_DIR / fonte["arquivo"]
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return careful_load_csv(caminho)

def careful_load_csv(path: Path, sep: str | None = None) -> pd.DataFrame:
    errors = []

    for encoding in encodings:
        for separator in separators:
            try:
                df = pd.read_csv(path, sep=separator, encoding=encoding, low_memory=False, na_values=na_values)

                if len(df.columns) <= 1:
                    errors.append(f"encoding={encoding}, sep={repr(separator)} → apenas {len(df.columns)} coluna(s)")
                    continue

                if has_replacement_char(df):
                    errors.append(f"encoding={encoding}, sep={repr(separator)} → caractere inválido nas colunas: {list(df.columns)}")
                    continue

                #print(f"[load] '{path.name}' lido com encoding={encoding}, sep={repr(separator)}")
                return df

            except UnicodeDecodeError as e:
                errors.append(f"encoding={encoding}, sep={repr(separator)} → UnicodeDecodeError: {e}")

            except pd.errors.ParserError as e:
                errors.append(f"encoding={encoding}, sep={repr(separator)} → ParserError: {e}")

            except Exception as e:
                errors.append(f"encoding={encoding}, sep={repr(separator)} → {type(e).__name__}: {e}")

    raise ValueError(f"Não foi possível ler '{path.name}' sem corromper colunas.\n"+ "\n".join(errors[-20:]))