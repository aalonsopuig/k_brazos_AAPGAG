import numpy as np
from algorithms.algorithm import Algorithm


class UCB1:
    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de ajuste que controla la exploración (usualmente c=1).
        """
        self.k = k
        self.c = c
        self.counts = np.zeros(k)  # Número de veces que se ha seleccionado cada brazo
        self.values = np.zeros(k)  # Estimación de la recompensa de cada brazo
        self.t = 0  # Número total de selecciones realizadas

    def select_arm(self) -> int:
        """ Selecciona el brazo basado en la fórmula de UCB1. """
        self.t += 1
        if 0 in self.counts:
            # Seleccionar cada brazo al menos una vez
            return np.argmin(self.counts)
        else:
            # Aplicar la ecuación de UCB1
            ucb_values = self.values + self.c * np.sqrt((2 * np.log(self.t)) / self.counts)
            return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float):
        """ Actualiza la estimación del valor del brazo seleccionado. """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] = ((n - 1) * self.values[chosen_arm] + reward) / n

    def reset(self):
        """ Reinicia el estado del algoritmo. """
        self.counts = np.zeros(self.k)
        self.values = np.zeros(self.k)
        self.t = 0