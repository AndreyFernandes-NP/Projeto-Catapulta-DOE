"""
supervised.py
-------------
Contém a classe Catapulta, sendo a classe principal do projeto, responsável por armazenar os datasets carregados e fornecer métodos para acessar esses dados.

Aprofundando, ela também contém métodos como o modelo supervisionado que será utilizado, função para ajuste de curvas e otimização, exploração breve de todos os dados armazenados, e etc.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.__config__ import PATHS

DATA_DIR = PATHS.data

class Catapulta:
    """Classe principal do projeto, responsável por servir como a pipeline principal para o fluxo de trabalho."""
    def __init__(self):
        self.fonts: list[str] = []
        self.datasets: dict[str, pd.DataFrame] = {}

    def get_fonts(self) -> list[str]:
        return self.fonts

    def store_dataset(self, nome: str, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            print(f"[Aviso] O dataset '{nome}' está vazio. Ele não será salvo.")
            return
        
        if nome in self.datasets:
            print(f"[Aviso] Substituindo dataset '{nome}' já existente.")

        self.datasets[nome] = dataframe

        if nome not in self.fonts:
            self.fonts.append(nome)
    
    def drop_empty_datasets(self) -> None:
        empty_fonts = [nome for nome, df in self.datasets.items() if df.empty]
        for nome in empty_fonts:
            print(f"[Aviso] Removendo dataset '{nome}' porque está vazio.")
            del self.datasets[nome]
            self.fonts.remove(nome)
    
    @property
    def parameters(self) -> np.ndarray:
        return self.get_parameters()

    def get_parameters(self, type='first') -> np.ndarray:
        if not self.datasets:
            print("[Aviso] Nenhum dataset armazenado. Parâmetros indisponíveis.")
            return np.array([])
        
        match type:
            case 'first':
                # Retorna os parâmetros do primeiro dataset encontrado
                for nome, df in self.datasets.items():
                    if not df.empty:
                        print(f"[Info] Parâmetros do dataset '{nome}': {len(df.columns)} colunas.")
                        return df.columns.values
            
            case 'last':
                # Retorna os parâmetros do último dataset encontrado
                for nome, df in reversed(self.datasets.items()):
                    if not df.empty:
                        print(f"[Info] Parâmetros do dataset '{nome}': {len(df.columns)} colunas.")
                        return df.columns.values
            
            case 'all':
                # Retorna a união de todos os parâmetros de todos os datasets
                all_columns = set()
                for nome, df in self.datasets.items():
                    if not df.empty:
                        all_columns.update(df.columns)

                print(f"[Info] Parâmetros combinados de todos os datasets: {len(all_columns)} colunas.")
                return np.array(list(all_columns))

            case 'intersection':
                # Retorna a interseção dos parâmetros de todos os datasets
                all_columns = [set(df.columns) for df in self.datasets.values() if not df.empty]
                if not all_columns:
                    print("[Aviso] Nenhum dataset válido encontrado. Parâmetros indisponíveis.")
                    return np.array([])

                common_columns = set.intersection(*all_columns)
                print(f"[Info] Parâmetros comuns de todos os datasets: {len(common_columns)} colunas.")
                return np.array(list(common_columns))

            case _:
                print(f"[Aviso] Tipo de parâmetro desconhecido: '{type}'. Use 'first', 'last', 'all' ou 'intersection'.")
                return np.array([])

        print("[Aviso] Nenhum dataset válido encontrado. Parâmetros indisponíveis.")
        return np.array([])