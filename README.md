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

El notebook **main.ipynb** es el punto de inicio del proyecto. Desde él, se configuran automáticamente todas las dependencias y se proporciona acceso a los notebooks de los experimentos.  

### ¿Cómo ejecutar el proyecto?  
Para poner en marcha la ejecución del proyecto, simplemente sigue estos pasos:  

1. **Abrir main.ipynb en Google Colab** utilizando el siguiente enlace: [Open in Colab](https://colab.research.google.com/github/aalonsopuig/k_brazos_AAPGAG/blob/main/main.ipynb)  

2. **Ejecutar todas las celdas** en orden automático:  
   En la barra de menú de Colab, haz clic en **Entorno de ejecución > Ejecutar todas**.  
   Esto iniciará una secuencia de pasos automática, que incluye:  
     - **Clonación del repositorio** con todo el código y los experimentos.  
     - **Instalación de todas las dependencias** necesarias para ejecutar los notebooks.  
     - **Carga de librerías esenciales** como NumPy, Matplotlib y SciPy.  
     - **Configuración del entorno** asegurando que todo esté listo para los experimentos.  

3. **Acceder a los notebooks de los experimentos**:  
   Al finalizar la ejecución, en la parte inferior del notebook principal, aparecerán enlaces directos a los notebooks individuales para cada método estudiado:  
     - **ε-Greedy**  
     - **Upper Confidence Bound (UCB)**  
     - **Ascenso de Gradiente**  
     - **Métodos Bayesianos**
       
   Basta con hacer clic en cualquier enlace para abrir y ejecutar el estudio correspondiente.  

### Reproducibilidad garantizada  
El proyecto está diseñado para que su ejecución en Colab sea completamente reproducible.  
Si ejecutas este notebook desde cero en un entorno nuevo, todos los experimentos funcionarán sin errores ni necesidad de configuraciones manuales.

## Tecnologías Utilizadas  
- **Lenguaje:** Python 3.x  
- **Bibliotecas:** NumPy, Matplotlib, Pandas, SciPy  
- **Entorno de ejecución:** Jupyter Notebook, Google Colab  
