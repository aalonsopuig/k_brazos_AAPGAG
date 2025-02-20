import numpy as np
from scipy.stats import beta
from algorithms.algorithm import Algorithm

class ThompsonSampling(Algorithm):
    """
    Implementación del algoritmo de Muestreo de Thompson para el problema de Bandit Multi-Brazo.
    Utiliza una distribución Beta para modelar la probabilidad de éxito de cada brazo.
    """
    
    def __init__(self, k: int, alpha: float = 1.0, beta: float = 1.0):
        """
        Inicializa el algoritmo de Muestreo de Thompson.
        
        :param k: Número de brazos del bandit.
        :param alpha: Parámetro inicial alfa para la distribución Beta de cada brazo.
        :param beta: Parámetro inicial beta para la distribución Beta de cada brazo.
        """
        super().__init__(k)
        self.alpha = np.ones(k) * alpha  # Vector con los valores iniciales de alfa para cada brazo
        self.beta = np.ones(k) * beta    # Vector con los valores iniciales de beta para cada brazo
    
    def select_arm(self) -> int:
        """
        Selecciona el brazo basado en la muestreo de la distribución Beta de cada brazo.
        La estrategia es muestrear cada brazo y seleccionar el que tenga la mayor muestra.
        
        :return: Índice del brazo seleccionado.
        """
        samples = np.random.beta(self.alpha, self.beta)  # Muestreo de cada brazo
        return np.argmax(samples)  # Selecciona el brazo con el mayor valor de muestreo
    
    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza los parámetros de la distribución Beta del brazo seleccionado en función de la recompensa obtenida.
        Si el brazo obtiene una recompensa de 1, se incrementa α. Si obtiene 0, se incrementa β.
        
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida (1 para éxito, 0 para fallo).
        """
        self.alpha[chosen_arm] += reward  # Incrementa alfa si hubo éxito
        self.beta[chosen_arm] += 1 - reward  # Incrementa beta si hubo fracaso


