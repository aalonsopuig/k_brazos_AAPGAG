"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Basado en código de Luis Daniel Hernández Molinero (um)
Date: 2025/02

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy

def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.
    
    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva con el nombre del algoritmo y sus parámetros.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    return label

def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.
    Muestra cómo evolucionan las recompensas promedio obtenidas por cada algoritmo a lo largo de los pasos de tiempo.
    
    :param steps: Número de pasos de tiempo.
    :type steps: int
    :param rewards: Matriz de recompensas promedio obtenidas por cada algoritmo en cada paso.
    :type rewards: np.ndarray
    :param algorithms: Lista de instancias de algoritmos a comparar.
    :type algorithms: List[Algorithm]
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)
    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.
    Esta métrica indica cuántas veces los algoritmos seleccionan el mejor brazo disponible.
    
    :param steps: Número de pasos de tiempo.
    :type steps: int
    :param optimal_selections: Matriz con el porcentaje de veces que se seleccionó el brazo óptimo en cada paso.
    :type optimal_selections: np.ndarray
    :param algorithms: Lista de instancias de algoritmos comparados.
    :type algorithms: List[Algorithm]
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx] * 100, label=label, linewidth=2)
    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Selección del Brazo Óptimo (%)', fontsize=14)
    plt.title('Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm]):
    """
    Genera histogramas de las recompensas promedio por brazo y el número de veces seleccionado.
    Muestra el desempeño de cada brazo y cuántas veces ha sido elegido por cada algoritmo.
    
    :param arm_stats: Lista de diccionarios con estadísticas de cada brazo para cada algoritmo.
    :type arm_stats: List[dict]
    :param algorithms: Lista de instancias de algoritmos comparados.
    :type algorithms: List[Algorithm]
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 7))
    
    for idx, algo in enumerate(algorithms):
        stats = arm_stats[idx]
        arms = list(range(len(stats['means'])))
        plt.bar(arms, stats['means'], alpha=0.6, label=get_algorithm_label(algo))
        for i, (mean, count) in enumerate(zip(stats['means'], stats['counts'])):
            plt.text(i, mean + 0.1, f"{count}", ha='center', fontsize=12)
    
    plt.xlabel('Índice del Brazo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Estadísticas de Selección de los Brazos', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.
    El regret mide la diferencia entre la recompensa obtenida y la mejor posible, acumulándose con el tiempo.
    
    :param steps: Número de pasos de tiempo.
    :type steps: int
    :param regret_accumulated: Matriz con la evolución del regret acumulado para cada algoritmo en cada paso.
    :type regret_accumulated: np.ndarray
    :param algorithms: Lista de instancias de algoritmos comparados.
    :type algorithms: List[Algorithm]
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)
    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title('Evolución del Regret Acumulado', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()
