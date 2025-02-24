# Práctica 1: Problema del Bandido de k-Brazos

## Información
- **Alumnos:** García Meroño, Andrés; Guillén Marquina, Pablo; Alonso Puig, Alejandro  
- **Asignatura:** Extensiones de Machine Learning  
- **Curso:** 2024/2025  
- **Grupo:** AAPGAG  

## Descripción
Este proyecto implementa y compara distintos algoritmos de solución para el problema del **bandido multi-brazo**.  
Se analizan estrategias clásicas y avanzadas, incluyendo **ε-greedy**, **UCB**, **Gradiente de Preferencias**,  
y métodos **bayesianos** como **Thompson Sampling** y **Bayesian UCB**.  

El objetivo es evaluar el rendimiento de cada algoritmo en términos de:
- **Recompensa promedio** a lo largo del tiempo.  
- **Selección del brazo óptimo** y su evolución.  
- **Estadísticas de cada brazo**, incluyendo su frecuencia de selección y recompensa media.  
- **Regret acumulado**, para medir la pérdida respecto a la estrategia óptima.  

### Modelado del Problema  
Los experimentos utilizan un **bandido multi-brazo con diferentes distribuciones de recompensa**,  
simulando entornos con diferentes niveles de incertidumbre:  

- **Bernoulli**: Cada brazo devuelve recompensa (1) con cierta probabilidad o fallo (0).  
- **Binomial**: Simula múltiples intentos con éxito/fallo en un número fijo de ensayos.  
- **Normal**: Modela entornos con recompensas continuas sujetas a ruido gaussiano.  
- **Beta**: Distribución flexible que representa incertidumbre en recompensas de [0,1].  

Cada algoritmo debe adaptarse a estas distribuciones para maximizar la recompensa acumulada.  

## Estructura  
El repositorio contiene los siguientes notebooks de análisis:  

- **`main.ipynb`**: Punto de entrada para ejecutar o acceder a los estudios desde **Google Colab**.  
- **`estudio_epsilon_greedy.ipynb`**: Evaluación de estrategias ε-greedy con diferentes valores de ε.  
- **`estudio_UCB.ipynb`**: Implementación y análisis de UCB clásico.  
- **`estudio_ascenso_gradiente.ipynb`**: Estudio del método de gradiente de preferencias.  
- **`estudio_bayesianos.ipynb`**: Comparación de **Thompson Sampling** y **Bayesian UCB** con distintos hiperparámetros.  
- **`requirements.txt`**: Dependencias necesarias para ejecutar los notebooks.  

## Instalación y Uso  
(Pendiente de completar)  

## Tecnologías Utilizadas  
- **Lenguaje:** Python 3.x  
- **Bibliotecas:** NumPy, Matplotlib, Pandas, SciPy  
- **Entorno de ejecución:** Jupyter Notebook, Google Colab  
