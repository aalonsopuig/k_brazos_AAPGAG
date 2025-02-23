import numpy as np
from scipy.stats import beta
from algorithms.algorithm import Algorithm

class BayesianUCB(Algorithm):
    """
    Implementación del algoritmo UCB Bayesiano para el problema de Bandit Multi-Brazo.
    Utiliza una distribución Beta y calcula un intervalo superior de confianza basado en percentiles
    en lugar de solo la media y la varianza.
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
        self.t = 0  # Contador de pasos

    def select_arm(self) -> int:
        """
        Selecciona el brazo basado en el cálculo del índice UCB Bayesiano.
        Usa el percentil 95 de la distribución Beta en lugar de solo media y varianza.

        :return: Índice del brazo seleccionado.
        """
        self.t += 1  # Incrementar el contador de iteraciones

        # Exploración forzada en los primeros pasos para asegurar cobertura inicial
        if np.sum(self.alpha + self.beta) < 10 * self.k:
            chosen_arm = np.random.choice(self.k)
            print(f"Paso {self.t}: Exploración forzada - seleccionando brazo {chosen_arm}")
            return chosen_arm

        # Cálculo del percentil 95% de la distribución Beta para cada brazo
        confidence_bounds = beta.ppf(0.95, self.alpha, self.beta)

        chosen_arm = np.argmax(confidence_bounds)  # Selecciona el brazo con mayor índice UCB
        print(f"Paso {self.t}: BayesianUCB seleccionó el brazo {chosen_arm}")
        
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza los parámetros de la distribución Beta del brazo seleccionado en función de la recompensa obtenida.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida (1 para éxito, 0 para fallo).
        """
        # Ajuste más suave de la penalización para evitar descartar brazos demasiado rápido
        self.alpha[chosen_arm] += reward  # Incrementa alpha si hubo éxito
        self.beta[chosen_arm] += 0.5 * (1 - reward)  # Penalización reducida para fracasos

        # Debugging: Mostrar actualización de parámetros
        print(f"Paso {self.t}: Update BayesianUCB")
        print(f"  Brazo {chosen_arm} - Recompensa: {reward}")
        print(f"  α (alpha) después de actualización: {self.alpha}")
        print(f"  β (beta) después de actualización: {self.beta}")
        print(f"  Conteo de selecciones: {self.counts}")

