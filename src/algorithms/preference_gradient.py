import numpy as np
from algorithms.algorithm import Algorithm

class PreferenceGradient(Algorithm):
    """
    Implementación del método de Gradiente de Preferencias para el problema del bandido multibrazo.
    
    Este método asigna una **preferencia** a cada brazo en lugar de mantener un promedio de recompensas.
    Luego, se convierte esta preferencia en una **probabilidad** de selección utilizando una política softmax.

    A diferencia de otros métodos (como epsilon-greedy o UCB), este enfoque utiliza **aprendizaje basado en gradientes**
    para actualizar sus preferencias en función de las recompensas recibidas.
    """

    def __init__(self, k: int, alpha: float = 0.1, baseline: bool = True):
        """
        Inicializa el algoritmo de Gradiente de Preferencias.

        :param k: Número de brazos disponibles en el bandido.
        :param alpha: Tasa de aprendizaje para actualizar las preferencias de los brazos.
        :param baseline: Si True, se usa una recompensa promedio como baseline para reducir la varianza.
        
        **Explicación de Parámetros:**
        - `alpha`: Controla qué tan rápido se ajustan las preferencias en función de las recompensas obtenidas.
        - `baseline`: Si se usa, el algoritmo resta la media de recompensas antes de actualizar las preferencias, lo que ayuda a estabilizar el aprendizaje.
        """

        super().__init__(k)
        self.alpha = alpha  # Tasa de aprendizaje para actualizar preferencias
        self.baseline = baseline  # Indica si se usa una media de recompensas como baseline
        self.preferences = np.zeros(k)  # Inicializa las preferencias de cada brazo a 0
        self.average_reward = 0  # Valor promedio de las recompensas obtenidas
        self.t = 0  # Contador de pasos

    def select_arm(self) -> int:
        """
        Selecciona un brazo utilizando la política Softmax sobre las preferencias.
        
        **Funcionamiento:**
        - Convierte las preferencias en probabilidades utilizando la función softmax.
        - Realiza un muestreo basado en estas probabilidades para seleccionar el brazo.

        :return: Índice del brazo seleccionado.
        """

        # Aplicar la transformación softmax a las preferencias de los brazos
        exp_preferences = np.exp(self.preferences)  # e^(H(a))
        self.action_probabilities = exp_preferences / np.sum(exp_preferences)  # Softmax: normaliza las probabilidades
        
        # Seleccionar un brazo según las probabilidades obtenidas
        chosen_arm = np.random.choice(self.k, p=self.action_probabilities)  
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las preferencias de los brazos en función de la recompensa obtenida.

        **Funcionamiento:**
        - Se ajustan las preferencias siguiendo la actualización del Gradiente Estocástico.
        - Se resta el baseline (si está activado) para mejorar estabilidad.
        - Se incrementa la preferencia del brazo seleccionado y se reducen proporcionalmente las demás.

        :param chosen_arm: Brazo que fue seleccionado.
        :param reward: Recompensa obtenida tras seleccionar el brazo.
        """

        self.t += 1  # Incrementar contador de pasos
        self.counts[chosen_arm] += 1  # Registrar selección del brazo

        # Actualizar el valor estimado de recompensa del brazo (media incremental)
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]

        # Calcular baseline si está activado (promedio de recompensas obtenidas hasta ahora)
        if self.baseline:
            self.average_reward += (reward - self.average_reward) / self.t

        # Usar la media de recompensas como baseline, o 0 si no está activado
        baseline_value = self.average_reward if self.baseline else 0

        # Aplicar la actualización del Gradiente Estocástico a cada brazo
        for a in range(self.k):
            if a == chosen_arm:
                # Incrementa la preferencia del brazo elegido basado en su recompensa
                self.preferences[a] += self.alpha * (reward - baseline_value) * (1 - self.action_probabilities[a])
            else:
                # Reduce proporcionalmente la preferencia de los demás brazos
                self.preferences[a] -= self.alpha * (reward - baseline_value) * self.action_probabilities[a]



    def reset(self):
        """
        Reinicia las preferencias y el promedio de recompensas para ejecutar nuevos experimentos.
        """
        self.preferences = np.zeros(self.k)  # Reiniciar todas las preferencias a 0
        self.average_reward = 0  # Reiniciar el baseline de recompensas
        self.t = 0  # Reiniciar el contador de pasos
