"""
visualization.py
----------------
Responsável pelas visualizações e análise exploratória
dos dados do experimento da catapulta.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


FEATURES = [
    "release_angle",
    "firing_angle",
    "cup_elevation",
    "pin_elevation",
    "bungee_elevation"
]

TARGET = "distancia"


def plot_correlacao(df: pd.DataFrame) -> None:
    """
    Exibe matriz de correlação entre variáveis.
    """

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Matriz de Correlação")

    plt.tight_layout()

    plt.show()


def plot_histogramas(df: pd.DataFrame) -> None:
    """
    Exibe histogramas de todas as variáveis numéricas.
    """

    df.hist(figsize=(12, 8))

    plt.tight_layout()

    plt.show()


def plot_scatterplots(df: pd.DataFrame) -> None:
    """
    Exibe scatterplots das variáveis de entrada
    contra a variável alvo.
    """

    for feature in FEATURES:

        plt.figure(figsize=(6, 4))

        sns.scatterplot(
            data=df,
            x=feature,
            y=TARGET
        )

        plt.title(f"{feature} vs {TARGET}")

        plt.tight_layout()

        plt.show()


def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Exibe boxplots das variáveis categóricas/discretas
    contra a variável alvo.
    """

    for feature in FEATURES:

        plt.figure(figsize=(6, 4))

        sns.boxplot(
            data=df,
            x=feature,
            y=TARGET
        )

        plt.title(f"Boxplot: {feature} vs {TARGET}")

        plt.tight_layout()

        plt.show()


def plot_interacoes(df: pd.DataFrame) -> None:
    """
    Visualiza possíveis interações entre fatores.
    """

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x="firing_angle",
        y=TARGET,
        hue="release_angle",
        s=80
    )

    plt.title("Interação: firing_angle x release_angle")

    plt.tight_layout()

    plt.show()


def plot_replicas(df: pd.DataFrame) -> None:
    """
    Analisa variabilidade entre réplicas.
    """

    plt.figure(figsize=(6, 4))

    sns.boxplot(
        data=df,
        x="replica",
        y=TARGET
    )

    plt.title("Variabilidade entre Réplicas")

    plt.tight_layout()

    plt.show()


def plot_top_resultados(
    df: pd.DataFrame,
    top_n: int = 10
) -> None:
    """
    Exibe os melhores resultados do experimento.
    """

    top = (
        df
        .sort_values(TARGET, ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=top,
        x="ensaio",
        y=TARGET
    )

    plt.title(f"Top {top_n} Ensaios")

    plt.tight_layout()

    plt.show()