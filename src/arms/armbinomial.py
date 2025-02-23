import numpy as np
from arms import Arm


class ArmBinomial(Arm):
    """
    Representa un brazo con distribución Binomial B(n, p).
    
    Un brazo con esta distribución devuelve un número de éxitos en n intentos,
    donde cada intento tiene una probabilidad p de éxito.
    
    Parámetros:
    - n: Número de ensayos en la distribución binomial (entero positivo).
    - p: Probabilidad de éxito en cada ensayo (0 <= p <= 1).
    """

    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de ensayos (debe ser mayor que 0).
        :param p: Probabilidad de éxito en cada ensayo (debe estar entre 0 y 1).
        """
        assert n > 0, "El número de ensayos (n) debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad de éxito (p) debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self) -> int:
        """
        Simula tirar del brazo y devuelve el número de éxitos obtenidos en n intentos.

        :return: Un valor entero que representa el número de éxitos.
        """
        return np.random.binomial(self.n, self.p)

    def get_expected_value(self) -> float:
        """
        Calcula el valor esperado de la distribución binomial, dado por E[X] = n * p.

        :return: Valor esperado de la distribución.
        """
        return self.n * self.p

    def __str__(self) -> str:
        """
        Devuelve una representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n_min: int = 1, n_max: int = 10, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con parámetros n y p aleatorios dentro de los rangos especificados.

        :param k: Número de brazos a generar.
        :param n_min: Valor mínimo de n.
        :param n_max: Valor máximo de n.
        :param p_min: Valor mínimo de p.
        :param p_max: Valor máximo de p.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n_min < n_max, "n_min debe ser menor que n_max."
        assert 0 <= p_min < p_max <= 1, "p_min y p_max deben estar entre 0 y 1, con p_min < p_max."

        arms = []
        for _ in range(k):
            n = np.random.randint(n_min, n_max + 1)  # Seleccionar n aleatorio en el rango
            p = np.random.uniform(p_min, p_max)  # Seleccionar p aleatorio en el rango
            p = round(p, 2)  # Redondear p a 2 decimales para mayor claridad
            arms.append(cls(n, p))

        return arms
