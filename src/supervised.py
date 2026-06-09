"""
supervised.py
-------------
Contém a classe Catapulta, sendo a classe principal do projeto, responsável por armazenar os datasets carregados e fornecer métodos para acessar esses dados.

Aprofundando, ela também contém métodos como o modelo supervisionado que será utilizado, função para ajuste de curvas e otimização, exploração breve de todos os dados armazenados, e etc.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.__config__ import PATHS

DATA_DIR = PATHS.data

class Catapulta:
    """Classe principal do projeto, responsável por servir como a pipeline principal para o fluxo de trabalho."""
    def __init__(self):
        self.fonts = []
        self.datasets = {}

    def get_fonts(self) -> list:
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