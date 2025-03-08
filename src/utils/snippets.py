#Snippets de código para incorporar a cada experimento para generar ficheros

# ASCENSO DE GRADIENTE ##########################################################
algoritmo="ascenso_gradiente"
from utils.experiment_data_saver import ExperimentDataSaver
# Construir etiquetas de los algoritmos basadas en su parámetro específico (epsilon, tau, c, etc.)
algorithm_labels = [f"Softmax (tau={algo.tau})" if hasattr(algo, "tau") 
                    else f"UCB1 (c={algo.c})" if hasattr(algo, "c")
                    else f"{type(algo).__name__}" for algo in algorithms]

# Crear el objeto para gestionar los archivos del experimento
data_saver = ExperimentDataSaver(algoritmo, algorithm_labels)

# Guardar los archivos CSV en una sola línea cada uno
data_saver.save_data(rewards, "rewards")
data_saver.save_data(optimal_selections, "optimal_selections")
data_saver.save_data(regret_accumulated, "regret")
data_saver.save_data(arm_stats, "arm_statistics", is_arm_statistics=True)


# BAYESIANOS ###################################################################
from utils.experiment_data_saver import ExperimentDataSaver

# Definir el nombre del experimento
algorithm = "bayesianos"

# Construcción de etiquetas para los algoritmos utilizados
algorithm_labels = [
    f"ThompsonSampling (α={algo.alpha[0]}, β={algo.beta[0]})" if isinstance(algo, ThompsonSampling)
    else f"BayesianUCB (γ={algo.gamma})" if isinstance(algo, BayesianUCB)
    else f"{type(algo).__name__}" for algo in algorithms
]

# Guardar los archivos CSV con los datos generados en el experimento
ExperimentDataSaver.generate_files(algorithm, algorithm_labels, rewards, optimal_selections, regret_accumulated, arm_stats)


# EPSILON-GREEDY ###################################################################
algoritmo = "epsilon_greedy"
from utils.experiment_data_saver import ExperimentDataSaver

# Construir etiquetas de los algoritmos basadas en el valor de epsilon
algorithm_labels = [f"EpsilonGreedy (epsilon={algo.epsilon})" for algo in algorithms]

# Crear el objeto para gestionar los archivos del experimento
data_saver = ExperimentDataSaver(algoritmo, algorithm_labels)

# Guardar los archivos CSV con los datos del experimento
data_saver.save_data(rewards, "rewards")  # Recompensa promedio
data_saver.save_data(optimal_selections, "optimal_selections")  # Selección del brazo óptimo
data_saver.save_data(regret_accumulated, "regret")  # Evolución del regret acumulado
data_saver.save_data(arm_stats, "arm_statistics", is_arm_statistics=True)  # Estadísticas de cada brazo



# UCB ###################################################################
algoritmo = "ucb"
from utils.experiment_data_saver import ExperimentDataSaver

# Construir etiquetas de los algoritmos basadas en sus parámetros específicos (c para UCB1 y alpha para UCB2)
algorithm_labels = [
    f"UCB1 (c={algo.c})" if isinstance(algo, UCB1) else 
    f"UCB2 (alpha={algo.alpha})" if isinstance(algo, UCB2) else 
    f"{type(algo).__name__}" 
    for algo in algorithms
]

# Crear el objeto para gestionar los archivos del experimento
data_saver = ExperimentDataSaver(algoritmo, algorithm_labels)

# Guardar los archivos CSV con los datos del experimento
data_saver.save_data(rewards, "rewards")  # Recompensa promedio
data_saver.save_data(optimal_selections, "optimal_selections")  # Selección del brazo óptimo
data_saver.save_data(regret_accumulated, "regret")  # Evolución del regret acumulado
data_saver.save_data(arm_stats, "arm_statistics", is_arm_statistics=True)  # Estadísticas de cada brazo

