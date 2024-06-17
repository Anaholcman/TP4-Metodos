import numpy as np
import matplotlib.pyplot as plt

# Paso 1: Generación de Datos
np.random.seed(0)  # Para reproducibilidad
n = 5
d = 100
A = np.random.randn(n, d)
b = np.random.randn(n)

# Paso 2: Definición de la Función de Costo
def F(x):
    return np.linalg.norm(A @ x - b) ** 2

def grad_F(x):
    return 2 * A.T @ (A @ x - b)

def F2(x, delta2):
    return F(x) + delta2 * np.linalg.norm(x) ** 2

def grad_F2(x, delta2):
    return grad_F(x) + 2 * delta2 * x

# Paso 3: Algoritmo de Gradiente Descendente
def gradient_descent(grad, x0, learning_rate, iterations, delta2=None):
    x = x0
    history = [x]
    for _ in range(iterations):
        if delta2 is not None:
            x = x - learning_rate * grad(x, delta2)
        else:
            x = x - learning_rate * grad(x)
        history.append(x)
    return x, history

# Parámetros
x0 = np.random.randn(d)
iterations = 1000
sigma_max = np.linalg.norm(A, ord=2)  # Valor singular máximo de A
lambda_max = np.linalg.eigvalsh(A.T @ A).max()  # Autovalor máximo de A^T A
s = 1 / lambda_max
delta2 = 1e-2 * sigma_max

# Minimización de F(x)
x_opt, history_F = gradient_descent(grad_F, x0, s, iterations)

# Minimización de F2(x)
x_opt_reg, history_F2 = gradient_descent(grad_F2, x0, s, iterations, delta2)

# Paso 4: Comparación con SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)
x_svd = VT.T @ np.linalg.pinv(np.diag(S)) @ U.T @ b

# Paso 5: Análisis de Resultados
# Evolución de la solución
history_F = np.array(history_F)
history_F2 = np.array(history_F2)

plt.plot(np.linalg.norm(history_F - x_opt, axis=1), label='F(x)')
plt.plot(np.linalg.norm(history_F2 - x_opt_reg, axis=1), label='F2(x)')
plt.yscale('log')
plt.xlabel('Iteraciones')
plt.ylabel('Norma de la diferencia')
plt.legend()
plt.show()

# Resultados
print("Solución por Gradiente Descendente para F(x):", x_opt)
print("Solución por Gradiente Descendente para F2(x):", x_opt_reg)
print("Solución por SVD:", x_svd)

# Análisis de Resultados
print("Norma de la diferencia (GD F(x) - SVD):", np.linalg.norm(x_opt - x_svd))
print("Norma de la diferencia (GD F2(x) - SVD):", np.linalg.norm(x_opt_reg - x_svd))
