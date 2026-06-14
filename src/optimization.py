"""
optimization.py
---------------
Responsável pela otimização dos parâmetros da catapulta utilizando scipy.optimize.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import (minimize, differential_evolution)

from mpl_toolkits.mplot3d import Axes3D


# Limites do experimento (bounds)

BOUNDS = [
    (156, 176),   # release_angle
    (90, 140),    # firing_angle
    (211, 290),   # cup_elevation
    (100, 200),   # pin_elevation
    (100, 200)    # bungee_elevation
]



def objective_function(
    x,
    model=None,
    poly=None
):
    """
    Função objetivo da otimização.

    ATUALMENTE PLACHOLDER!!!
    - depois ver de utilizar model.predict()
    """

    # Apenas para testes estruturais
    # Simula uma função com máximo próximo do centro

    release_angle = x[0]
    firing_angle = x[1]
    cup_elevation = x[2]
    pin_elevation = x[3]
    bungee_elevation = x[4]

    simulated_distance = (
        - (release_angle - 170) ** 2
        - (firing_angle - 120) ** 2
        - 0.01 * (cup_elevation - 260) ** 2
        - 0.01 * (pin_elevation - 170) ** 2
        - 0.01 * (bungee_elevation - 180) ** 2
        + 300
    )

    # minimize() minimiza
    # queremos maximizar distância
    return -simulated_distance


# Otimização local

def optimize_local():
    """
    Executa otimização local usando minimize().
    """

    initial_guess = [
        166,
        115,
        250,
        150,
        150
    ]

    result = minimize(
        fun=objective_function,
        x0=initial_guess,
        bounds=BOUNDS
    )

    return result


# Otimização global

def optimize_global():
    """
    Executa otimização global usando differential_evolution().
    """

    result = differential_evolution(
        func=objective_function,
        bounds=BOUNDS
    )

    return result



# Visualização 2D

def plot_2d_cut():
    """
    Gera corte 2D da função objetivo.

    Varia:
    - firing_angle

    Mantém:
    - demais variáveis fixas
    """

    firing_values = np.linspace(90, 140, 100)

    distances = []

    for firing in firing_values:

        x = [
            170,
            firing,
            260,
            170,
            180
        ]

        dist = -objective_function(x)

        distances.append(dist)

    plt.figure(figsize=(8, 5))

    plt.plot(
        firing_values,
        distances
    )

    plt.xlabel("firing_angle")

    plt.ylabel("distancia estimada")

    plt.title("Corte 2D da Função Estimada")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# Visuzalização 3D

def plot_3d_surface():
    """
    Gera superfície 3D da função estimada.

    Variáveis:
    - release_angle
    - firing_angle
    """

    release = np.linspace(156, 176, 50)

    firing = np.linspace(90, 140, 50)

    X, Y = np.meshgrid(
        release,
        firing
    )

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            x = [
                X[i, j],
                Y[i, j],
                260,
                170,
                180
            ]

            Z[i, j] = -objective_function(x)

    fig = plt.figure(figsize=(10, 7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot_surface(
        X,
        Y,
        Z
    )

    ax.set_xlabel("release_angle")

    ax.set_ylabel("firing_angle")

    ax.set_zlabel("distancia")

    ax.set_title("Superfície 3D da Função Estimada")

    plt.tight_layout()

    plt.show()


# EXECUÇÃO DE TESTE

if __name__ == "__main__":

    print("=" * 50)
    print("OTIMIZAÇÃO LOCAL")
    print("=" * 50)

    local_result = optimize_local()

    print(local_result)

    print()

    print("=" * 50)
    print("OTIMIZAÇÃO GLOBAL")
    print("=" * 50)

    global_result = optimize_global()

    print(global_result)

    plot_2d_cut()

    plot_3d_surface()