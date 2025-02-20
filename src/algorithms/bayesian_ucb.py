import numpy as np
from scipy.stats import beta
from algorithms.algorithm import Algorithm

class BayesianUCB(Algorithm):
    """
    Implementación del algoritmo UCB Bayesiano para el problema de Bandit Multi-Brazo.
    Utiliza una distribución Beta y calcula un intervalo superior de confianza basado en la desviación estándar.
    """
    
    def __init__(self, k: int, gamma: float = 2.0):
        """
        Inicializa el algoritmo UCB Bayesiano.
        
        :param k: Número de brazos del bandit.
        :param gamma: Factor de ajuste para la exploración (número de desviaciones estándar a considerar).
        """
        super().__init__(k)
        self.alpha = np.ones(k)  # Inicializa alfa en 1 para cada brazo
        self.beta = np.ones(k)   # Inicializa beta en 1 para cada brazo
        self.gamma = gamma  # Factor de ajuste para la exploración
    
    def select_arm(self) -> int:
        """
        Selecciona el brazo basado en el cálculo del índice UCB Bayesiano.
        La estrategia usa la media y la varianza de la distribución Beta de cada brazo.
        
        :return: Índice del brazo seleccionado.
        """
        mean_estimates = self.alpha / (self.alpha + self.beta)  # Media esperada de cada brazo
        variance_estimates = (self.alpha * self.beta) / (((self.alpha + self.beta) ** 2) * (self.alpha + self.beta + 1))
        confidence_bounds = mean_estimates + self.gamma * np.sqrt(variance_estimates)  # Índice UCB
        return np.argmax(confidence_bounds)  # Selecciona el brazo con mayor índice UCB
    
    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza los parámetros de la distribución Beta del brazo seleccionado en función de la recompensa obtenida.
        
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida (1 para éxito, 0 para fallo).
        """
        self.alpha[chosen_arm] += reward  # Incrementa alfa si hubo éxito
        self.beta[chosen_arm] += 1 - reward  # Incrementa beta si hubo fracaso
