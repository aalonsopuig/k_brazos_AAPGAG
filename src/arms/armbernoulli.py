import numpy as np
from arms import Arm

class ArmBernoulli(Arm):
    """
    Clase que modela un brazo (arm) con una distribución Bernoulli.
    
    Cada vez que se tira de este brazo, se obtiene una recompensa de 1 con 
    una probabilidad `p`, y una recompensa de 0 con probabilidad `1 - p`.
    
    Métodos principales:
    - pull(): Devuelve 1 con probabilidad `p` y 0 con `1 - p`.
    - get_expected_value(): Retorna la probabilidad de éxito `p`, que es el valor esperado.
    - generate_arms(k, p_min, p_max): Genera `k` brazos con probabilidades aleatorias en el rango especificado.
    """
    
    def __init__(self, p: float):
        """
        Inicializa un brazo con distribución Bernoulli.

        :param p: Probabilidad de éxito (0 ≤ p ≤ 1).
        """
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0, 1]."
        self.p = p  # Probabilidad de obtener recompensa 1

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Bernoulli.

        :return: 1 con probabilidad p, 0 con probabilidad (1 - p).
        """
        return np.random.binomial(1, self.p)  # Simula un éxito (1) o un fracaso (0)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.

        :return: Probabilidad de éxito (p).
        """
        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción detallada del brazo Bernoulli.
        """
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Probabilidad mínima de éxito.
        :param p_max: Probabilidad máxima de éxito.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "Las probabilidades deben estar en el rango [0, 1] y p_min < p_max."

        # Generar k valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)

        arms = [ArmBernoulli(p) for p in p_values]
        return arms
