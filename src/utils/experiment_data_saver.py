import pandas as pd
import numpy as np
import csv
import os

class ExperimentDataSaver:
    """
    Clase para gestionar la generación y almacenamiento de datos de experimentos en archivos CSV.
    Permite guardar datos de recompensas, selecciones óptimas, regret acumulado y estadísticas de brazos.
    """

    def __init__(self, experiment_name: str, algorithm_labels: list, sample_rate: int = 20):
        """
        Inicializa el gestor de almacenamiento de datos.

        :param experiment_name: Nombre del experimento (se usará como prefijo en los archivos CSV).
        :param algorithm_labels: Lista con los nombres de los algoritmos para etiquetar las columnas.
        :param sample_rate: Frecuencia de muestreo en los pasos (por defecto, cada 20 pasos).
        """
        self.experiment_name = experiment_name
        self.algorithm_labels = algorithm_labels
        self.sample_rate = sample_rate
        self.data_path = "data"

        # Crear el directorio si no existe
        os.makedirs(self.data_path, exist_ok=True)

    def save_data(self, data, data_type: str, is_arm_statistics: bool = False):
        """
        Guarda los datos en un archivo CSV con muestreo adecuado.

        :param data: Matriz de datos (rewards, optimal selections, regret) o estadísticas de brazos.
        :param data_type: Tipo de dato a guardar ("rewards", "optimal_selections", "regret", "arm_statistics").
        :param is_arm_statistics: Indica si los datos corresponden a estadísticas de brazos.
        """
        file_name = f"{self.data_path}/{self.experiment_name}_{data_type}_sampled.csv"

        if is_arm_statistics:
            # Guardar estadísticas de brazos
            with open(file_name, mode='w', newline='', encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Algorithm", "Arm", "Mean_Reward", "Count"])
                for idx, algo in enumerate(data):  # `data` es `arm_stats`
                    for arm_index, (mean, count) in enumerate(zip(algo["means"], algo["counts"])):
                        writer.writerow([self.algorithm_labels[idx], arm_index, mean, count])

        else:
            # Guardar datos estructurados (Recompensa, Selecciones Óptimas, Regret)
            steps_sampled = np.arange(0, data.shape[1], self.sample_rate)
            df_sampled = pd.DataFrame(data[:, ::self.sample_rate].T, columns=self.algorithm_labels, index=steps_sampled)
            df_sampled.to_csv(file_name, index_label="Step", float_format="%.2f")

        print(f"Archivo guardado: {file_name}")


