import numpy as np
from algorithms.algorithm import Algorithm

class BayesianUCB(Algorithm):
    """
    Implementación del algoritmo UCB Bayesiano para el problema de Bandit Multi-Brazo.
    Utiliza una distribución Beta para modelar la incertidumbre en la recompensa de cada brazo.
    """

    def __init__(self, k: int, gamma: float = 2.0):
        """
        Inicializa el algoritmo UCB Bayesiano.

        :param k: Número de brazos del bandit.
        :param gamma: Factor de ajuste para la exploración.
        """
        super().__init__(k)
        self.alpha = np.ones(k)  # Inicializa alfa en 1 para cada brazo
        self.beta = np.ones(k)   # Inicializa beta en 1 para cada brazo
        self.gamma = gamma  # Factor de exploración
        self.values = np.zeros(k)  # Recompensa promedio estimada de cada brazo

    def select_arm(self) -> int:
        """
        Selecciona el brazo con el valor UCB más alto.

        :return: Índice del brazo seleccionado.
        """
        # Calcular media de la distribución Beta
        mean_estimates = self.alpha / (self.alpha + self.beta)

        # Calcular desviación estándar de la distribución Beta
        variance_estimates = (self.alpha * self.beta) / (((self.alpha + self.beta) ** 2) * (self.alpha + self.beta + 1))
        std_estimates = np.sqrt(variance_estimates)

        # Calcular UCB usando la ecuación de la diapositiva
        ucb_values = mean_estimates + self.gamma * std_estimates

        # Seleccionar el brazo con el mayor UCB
        return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza los parámetros de la distribución Beta del brazo seleccionado.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida (1 para éxito, 0 para fallo).
        """
        # Actualizar la distribución Beta del brazo seleccionado
        self.alpha[chosen_arm] += reward  # Incrementar alfa si hubo éxito
        self.beta[chosen_arm] += (1 - reward)  # Incrementar beta si hubo fracaso

        # Actualizar la recompensa promedio del brazo elegido
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]
