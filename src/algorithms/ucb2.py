import numpy as np
from algorithms.algorithm import Algorithm


class UCB2:
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2.

        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste para el balance entre exploración y explotación.
        """
        self.k = k
        self.alpha = alpha
        self.counts = np.zeros(k)  # Número de veces que se ha seleccionado cada brazo
        self.values = np.zeros(k)  # Estimación de la recompensa de cada brazo
        self.t = 0  # Número total de selecciones realizadas
        self.epochs = np.zeros(k)  # Número de épocas para cada brazo

    def select_arm(self) -> int:
        """ Selecciona el brazo basado en la fórmula de UCB2. """
        self.t += 1
        if 0 in self.counts:
            # Seleccionar cada brazo al menos una vez
            return np.argmin(self.counts)
        else:
            # Aplicar la ecuación de UCB2
            tau_k = np.ceil((1 + self.alpha) ** self.epochs)
            ucb_values = self.values + np.sqrt(((1 + self.alpha) * np.log(np.e * self.t / tau_k)) / (2 * tau_k))
            return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float):
        """ Actualiza la estimación del valor del brazo seleccionado. """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] = ((n - 1) * self.values[chosen_arm] + reward) / n
        self.epochs[chosen_arm] += 1

    def reset(self):
        """ Reinicia el estado del algoritmo. """
        self.counts = np.zeros(self.k)
        self.values = np.zeros(self.k)
        self.t = 0
        self.epochs = np.zeros(self.k)
