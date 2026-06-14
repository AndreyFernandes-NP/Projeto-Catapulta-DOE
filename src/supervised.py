"""
supervised.py
-------------
Contém a classe Catapulta, sendo a classe principal do projeto, responsável por armazenar os datasets carregados e fornecer métodos para acessar esses dados.

Aprofundando, ela também contém métodos como o modelo supervisionado que será utilizado, função para ajuste de curvas e otimização, exploração breve de todos os dados armazenados, e etc.
"""

import sys
from pathlib import Path

from typing import Any
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.__config__ import PATHS
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

DATA_DIR = PATHS.data

class Catapulta:
    """Classe principal do projeto, responsável por servir como a pipeline principal para o fluxo de trabalho."""
    def __init__(self):
        self.fonts: list[str] = []
        self.datasets: dict[str, pd.DataFrame] = {}
        self.polynomials: dict[str, np.ndarray] = {}

        self.features: list[str] = []
        self.model: Any = None
        # As métricas e predições são armazenadas como listas para permitir comparações da eficácia do mesmo modelo entre diferentes dataframes
        self.metrics: list[dict[str, Any]] = []
        self.predictions: list[pd.DataFrame] | None = None

    # TODO: Se estiver se sentindo romântico, dá pra fazer (model) como parâmetro e fazer um self.model = model(**hyperparameters)
    # Só é preciso fazer o import dos modelos no __main__ (ou onde for chamar a função) ao invés daqui.
    def build_model(self, **hyperparameters) -> bool:
        try:
            self.model = LinearRegression(**hyperparameters)
        except Exception as e:
            print(f"[Erro] Ocorreu uma falha durante a construção do modelo: {e}")
            return False
        
        return True
    
    def create_polynomial(self, df: pd.DataFrame, degree: int) -> np.ndarray | None:
        try:
            poly = PolynomialFeatures(degree)
            df_adj = poly.fit_transform(df.copy())
            return df_adj
        
        except Exception as e:
            print(f"[Erro] Ocorreu uma falha na tentativa de criação do polimônio: {e}")

        return None

    def store_dataset(self, nome: str, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            print(f"[Aviso] O dataset '{nome}' está vazio. Ele não será salvo.")
            return
        
        if nome in self.datasets:
            print(f"[Aviso] Substituindo dataset '{nome}' já existente.")

        self.datasets[nome] = dataframe

        if nome not in self.fonts:
            self.fonts.append(nome)
    
    def store_polynomial(self, nome: str, poly: np.ndarray | None) -> None:
        if poly is None or poly.size == 0:
            print(f"[Aviso] O polinômio '{nome}' está vazio. Ele não será salvo.")
            return
        
        if nome in self.polynomials:
            print(f"[Aviso] Substituindo polinômio '{nome}' já existente.")
        
        self.polynomials[nome] = poly
    
    def get_fonts(self) -> list[str]:
        return self.fonts
    
    def get_dataset(self, nome: str) -> pd.DataFrame | None:
        if nome in self.datasets:
            return self.datasets[nome]
        else:
            print(f"[Aviso] Dataset '{nome}' não encontrado na lista de datasets.")
        return None
    
    def get_polynomial(self, nome: str, **poly_params) -> np.ndarray | None:
        if nome in self.polynomials:
            return self.polynomials[nome]
        else:
            print(f"[Aviso] Polinômio do dataset '{nome}' não encontrado na lista de polinômios, tentando criação.")
            poly = self.create_polynomial(**poly_params) if poly_params else None

            if poly is not None:
                self.store_polynomial(nome, poly)
                return poly
            else:
                if not poly_params: print(f"[Aviso] Não foi possível criar o polinômio para '{nome}', a função não recebeu parâmetros para tal.")

        return None
    
    def drop_empty_datasets(self) -> None:
        empty_fonts = [nome for nome, df in self.datasets.items() if df.empty]
        for nome in empty_fonts:
            print(f"[Aviso] Removendo dataset '{nome}' porque está vazio.")
            del self.datasets[nome]
            self.fonts.remove(nome)
    
    def load_df(self, nome: str) -> pd.DataFrame:
        if nome not in self.datasets:
            print(f"[Erro] Dataset '{nome}' não encontrado.")
            return pd.DataFrame()

        return self.datasets[nome]
    
    def analyze_datasets(self) -> None:
        if not self.datasets:
            print("[Aviso] Nenhum dataset armazenado para análise.")
            return
        
        for nome, df in self.datasets.items():
            if df.empty:
                print(f"[Aviso] O dataset '{nome}' está vazio. Pulando análise.")
                continue
            
            print(f"Análise do dataset '{nome}':")
            print(f"- Número de linhas: {len(df)}")
            print(f"- Número de colunas: {len(df.columns)}")
            print(f"- Colunas: {list(df.columns)}")
            print(f"- Tipos de dados:\n{df.dtypes}")
            print(f"- Estatísticas descritivas:\n{df.describe(include='all')}")
            print()
    
    def curve_adjustment(self, nome: str, df: pd.DataFrame, poly_degree: int) -> bool:
        print(f"[Execução] Criando polinômio para '{nome}'...")
        df_adj = self.create_polynomial(df, poly_degree)
        self.store_polynomial(nome, df_adj)

        print(f"[Execução] Polinômio de '{nome}' criado e salvo diretamente.")
        return True
    
    @property
    def parameters(self) -> np.ndarray:
        return self.get_parameters()

    def get_parameters(self, type: str ='first') -> np.ndarray:
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