import numpy as np
from scipy.stats import beta
from algorithms.algorithm import Algorithm

class ThompsonSampling(Algorithm):
    """
    Implementación del algoritmo de Muestreo de Thompson para el problema de Bandit Multi-Brazo.
    Utiliza una distribución Beta para modelar la probabilidad de éxito de cada brazo y selecciona 
    acciones de manera probabilística en función de estas creencias.
    """

    def __init__(self, k: int, alpha: float = 1.0, beta: float = 1.0):
        """
        Inicializa el algoritmo de Muestreo de Thompson.

        :param k: Número de brazos del bandit.
        :param alpha: Parámetro inicial alfa para la distribución Beta de cada brazo (prior de éxito).
        :param beta: Parámetro inicial beta para la distribución Beta de cada brazo (prior de fracaso).
        """
        super().__init__(k)
        self.alpha = np.ones(k) * alpha  # Inicializa α para cada brazo (número de éxitos iniciales)
        self.beta = np.ones(k) * beta    # Inicializa β para cada brazo (número de fracasos iniciales)
        self.values = np.zeros(k)  # Almacena la recompensa promedio de cada brazo (para visualización)
        self.counts = np.zeros(k, dtype=int)  # Contador de selecciones por brazo
        self.t = 0  # Contador de pasos en la ejecución

    def select_arm(self) -> int:
        """
        Selecciona un brazo usando muestreo de la distribución Beta.
        
        **Proceso:**
        1. Para cada brazo k, se obtiene un valor `θ_k` muestreando Beta(α_k, β_k).
        2. Se selecciona el brazo con el mayor valor `θ_k`, que corresponde a la máxima probabilidad muestreada.
        
        :return: Índice del brazo seleccionado.
        """
        self.t += 1  # Incrementar contador de iteraciones
        samples = np.random.beta(self.alpha, self.beta, size=self.k)  # Muestreo de cada brazo
        return np.argmax(samples)  # Selecciona el brazo con el mayor valor de muestreo

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza los parámetros de la distribución Beta del brazo seleccionado en función de la recompensa obtenida.

        **Proceso:**
        - Si el brazo seleccionado tuvo éxito (reward=1), se incrementa α.
        - Si el brazo seleccionado falló (reward=0), se incrementa β.
        
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida (1 para éxito, 0 para fallo).
        """
        self.alpha[chosen_arm] += reward  # Incrementa α si hubo éxito
        self.beta[chosen_arm] += 1 - reward  # Incrementa β si hubo fracaso

        # Actualiza la recompensa promedio del brazo (solo para visualización y análisis)
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]

    def reset(self):
        """
        Reinicia los valores de α, β y las estadísticas del experimento.
        """
        self.alpha = np.ones(self.k)  # Reiniciar α para cada brazo
        self.beta = np.ones(self.k)   # Reiniciar β para cada brazo
        self.counts = np.zeros(self.k)  # Reiniciar contador de selecciones
        self.values = np.zeros(self.k)  # Reiniciar recompensa promedio
        self.t = 0  # Reiniciar contador de pasos
