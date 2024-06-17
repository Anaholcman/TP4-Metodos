import numpy as np
import matplotlib.pyplot as plt

# Funciones de Costo
def cost_function(A, x, b):
    return np.dot((np.dot(A, x) - b).T, (np.dot(A, x) - b))

def cost_function_regularized(A, x, b, delta):
    return cost_function(A, x, b) + delta * np.linalg.norm(x)**2

# Gradientes
def gradient(A, x, b):
    return 2 * np.dot(A.T, (np.dot(A, x) - b))

def gradient_regularized(A, x, b, delta):
    return gradient(A, x, b) + 2 * delta * x

# Algoritmos de Gradiente Descendente
def gradient_descent(A, b, s, max_iter=1000):
    x = np.random.rand(A.shape[1])  # Condición inicial aleatoria
    errors = []
    for _ in range(max_iter):
        error = cost_function(A, x, b)
        errors.append(error)
        grad = gradient(A, x, b)
        x = x - s * grad
    return x, errors

def gradient_descent_regularized(A, b, s, delta, max_iter=1000):
    x = np.random.rand(A.shape[1])  # Condición inicial aleatoria
    errors = []
    for _ in range(max_iter):
        error = cost_function_regularized(A, x, b, delta)
        errors.append(error)
        grad = gradient_regularized(A, x, b, delta)
        x = x - s * grad
    return x, errors

# Generación de Datos Aleatorios
n, d = 5, 100
A = np.random.rand(n, d)
b = np.random.rand(n)

# Cálculo de sigma_max y lambda_max
sigma_max = np.linalg.svd(A, compute_uv=False)[0]
lambda_max = np.linalg.norm(A, ord=2)**2

# Parámetros
s = 1 / lambda_max
delta = 10**(-2) * sigma_max

# Ejecutar el algoritmo de gradiente descendente para F(x)
x_solution, errors = gradient_descent(A, b, s)

# Ejecutar el algoritmo de gradiente descendente para F2(x) con regularización L2
x_solution_reg, errors_reg = gradient_descent_regularized(A, b, s, delta)

# Comparar con la solución obtenida mediante SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)
x_svd = np.dot(VT.T, np.dot(np.linalg.inv(np.diag(S)), np.dot(U.T, b)))

# Graficar la evolución del error
plt.plot(range(len(errors)), errors, label='F(x)')
plt.plot(range(len(errors_reg)), errors_reg, label='F2(x) con Regularización')
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.title('Evolución del Error')
plt.legend()
plt.show()

# Graficar la comparación de soluciones
plt.plot(x_solution, label='Gradiente Descendente F(x)')
plt.plot(x_solution_reg, label='Gradiente Descendente F2(x) con Regularización', linestyle='dashed')
plt.plot(x_svd, label='SVD', linestyle='dotted')
plt.xlabel('Índice de Componente')
plt.ylabel('Valor de Componente')
plt.title('Comparación de Soluciones')
plt.legend()
plt.show()

# Análisis del impacto de la regularización L2
delta_values = [10**(-2) * sigma_max, 10**(-1) * sigma_max, 10**(0) * sigma_max]

for delta in delta_values:
    x_solution_reg, errors_reg = gradient_descent_regularized(A, b, s, delta)
    plt.plot(range(len(errors_reg)), errors_reg, label=f'δ²={delta}')
    
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.title('Impacto de la Regularización L2')
plt.legend()
plt.show()
