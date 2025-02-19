import numpy as np
from arms import Arm

class ArmBeta(Arm):
    """
    Representa un brazo con una distribución Beta B(α, β).

    Se usa principalmente en el Muestreo de Thompson para modelar la probabilidad de éxito basada en experiencias previas.
    """

    def __init__(self, alpha: float, beta: float):
        """
        Inicializa el brazo con distribución Beta.

        :param alpha: Parámetro α (shape) de la distribución Beta (debe ser > 0).
        :param beta: Parámetro β (shape) de la distribución Beta (debe ser > 0).
        """
        assert alpha > 0 and beta > 0, "Los parámetros alpha y beta deben ser positivos."

        self.alpha = alpha
        self.beta = beta

    def pull(self) -> float:
        """
        Genera una recompensa siguiendo una distribución Beta B(α, β).

        :return: Recompensa obtenida del brazo (un valor en [0,1]).
        """
        reward = np.random.beta(self.alpha, self.beta)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Beta.

        :return: Valor esperado de la distribución: E[X] = α / (α + β)
        """
        return self.alpha / (self.alpha + self.beta)

    def __str__(self):
        """
        Representación en cadena del brazo Beta.

        :return: Descripción detallada del brazo Beta.
        """
        return f"ArmBeta(alpha={self.alpha}, beta={self.beta})"

    @classmethod
    def generate_arms(cls, k: int, alpha_min: float = 1.0, alpha_max: float = 5.0, beta_min: float = 1.0, beta_max: float = 5.0):
        """
        Genera k brazos con parámetros α y β aleatorios dentro de un rango definido.

        :param k: Número de brazos a generar.
        :param alpha_min: Valor mínimo de α.
        :param alpha_max: Valor máximo de α.
        :param beta_min: Valor mínimo de β.
        :param beta_max: Valor máximo de β.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert alpha_min < alpha_max and beta_min < beta_max, "Los rangos de alpha y beta deben ser válidos."

        arms = []
        for _ in range(k):
            alpha = np.round(np.random.uniform(alpha_min, alpha_max), 2)
            beta = np.round(np.random.uniform(beta_min, beta_max), 2)
            arms.append(cls(alpha, beta))

        return arms
