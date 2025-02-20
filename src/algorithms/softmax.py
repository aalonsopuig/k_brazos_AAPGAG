import numpy as np
from algorithms.algorithm import Algorithm

class Softmax(Algorithm):
    """
    Implementación del método Softmax para el problema del bandido multibrazo.

    En este enfoque, en lugar de elegir siempre el brazo con la recompensa esperada más alta
    (como en epsilon-greedy o UCB), se asigna una probabilidad de selección proporcional a su valor esperado
    utilizando la función **softmax**.

    Cuanto mayor sea la recompensa promedio de un brazo, mayor será la probabilidad de que sea seleccionado,
    pero todos los brazos tienen cierta probabilidad de ser elegidos.
    """

    def __init__(self, k: int, tau: float = 1.0):
        """
        Inicializa el algoritmo Softmax.

        :param k: Número de brazos disponibles en el bandido.
        :param tau: Parámetro de temperatura que controla el grado de exploración.
        
        **Explicación de Parámetros:**
        - `tau` (temperatura): Cuanto mayor sea, más exploración habrá (las probabilidades de selección serán más uniformes).
          Si `tau` es muy bajo, el algoritmo se comportará de manera más codiciosa, eligiendo casi siempre el mejor brazo.
        """

        super().__init__(k)
        assert tau > 0, "El parámetro tau debe ser mayor que 0."
        
        self.tau = tau  # Temperatura para la función softmax

    def select_arm(self) -> int:
        """
        Selecciona un brazo utilizando la política Softmax.

        **Funcionamiento:**
        - Aplica la función Softmax a los valores estimados de recompensa.
        - Utiliza las probabilidades obtenidas para hacer una selección ponderada.

        :return: Índice del brazo seleccionado.
        """

        # Aplicar softmax a las estimaciones de recompensa de los brazos
        exp_values = np.exp(self.values / self.tau)  # e^(Q(a)/tau)
        probabilities = exp_values / np.sum(exp_values)  # Normalización para obtener probabilidades

        # Seleccionar un brazo basado en la distribución de probabilidades
        chosen_arm = np.random.choice(self.k, p=probabilities)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza la estimación de recompensa del brazo seleccionado.

        **Funcionamiento:**
        - Ajusta el valor esperado del brazo elegido usando un promedio incremental.
        - No afecta directamente a los demás brazos.

        :param chosen_arm: Brazo que fue seleccionado.
        :param reward: Recompensa obtenida tras seleccionar el brazo.
        """

        # Actualización incremental del valor estimado del brazo elegido
        self.counts[chosen_arm] += 1  # Incrementar el contador de veces seleccionado
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]

    def reset(self):
        """
        Reinicia los valores de recompensa estimada para ejecutar nuevos experimentos.
        """
        super().reset()  # Llama a la función reset de la clase padre
