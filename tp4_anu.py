import numpy as np
import matplotlib.pyplot as plt

# Generar matriz A y vector b aleatorios
np.random.seed(0)  # Para reproducibilidad
A = np.random.rand(5, 100)
b = np.random.rand(5, 1)

# Vector inicial x con distribución uniforme [0, 1]
x = np.random.rand(100, 1)

# Definir función de costo F(x)
def F(x):
    return np.transpose(A @ x - b) @ (A @ x - b)

# Definir función de costo F2(x)
delta = 0.01 * np.sqrt(np.max(np.linalg.eigvals(2 * np.transpose(A) @ A)))

def F2(x):
    return F(x) + delta * (np.linalg.norm(x)**2)

# Calcular gradiente de F
def grad_F(x):
    return 2 * np.transpose(A) @ (A @ x - b)

# Calcular gradiente de F2
def grad_F2(x):
    return grad_F(x) + 2 * delta * x

# Algoritmo de gradiente descendente
def gradient_descent(A, b, x, num_iterations, delta):
    s = 1 / np.max(np.linalg.eigvals(2 * np.transpose(A) @ A))
    F_values = []
    F2_values = []
    norm_x_values = []
    errors = []

    for i in range(num_iterations):
        x = x - s * grad_F(x)
        F_values.append(F(x).item())  # Convertir a escalar
        F2_values.append(F2(x).item())  # Convertir a escalar
        norm_x_values.append(np.linalg.norm(x).item())  # Convertir a escalar
        errors.append(np.linalg.norm(A @ x - b).item())  # Convertir a escalar
    
    return x, F_values, F2_values, norm_x_values, errors

# Realizar las iteraciones
num_iterations = 1000
x_final, F_values, F2_values, norm_x_values, errors = gradient_descent(A, b, x, num_iterations, delta)

# Graficar los resultados


# Gráfico de la norma 2 de x en función de iteraciones
plt.plot(norm_x_values, label='Norma 2 de x', color='darkgreen')
plt.xlabel('Iteraciones')
plt.ylabel('Norma 2 de x')
plt.title('Norma 2 de x en función de iteraciones')
plt.show()
# Gráfico del error ||Ax - b|| en función de iteraciones

plt.plot(errors, label='Error de gradiente descendente', color='darkgreen', linewidth=1.5)
# Para comparar con el error usando SVD
x_svd = np.linalg.lstsq(A, b, rcond=None)[0]
error_svd = np.linalg.norm(A @ x_svd - b)
plt.axhline(y=error_svd, color='purple', linestyle='--', label='Error de SVD', linewidth=1.5)
plt.xlabel('Iteraciones')
plt.ylabel('Error ||Ax - b||')
plt.legend()
plt.yscale('log')
plt.title('Error ||Ax - b|| en función de iteraciones')

plt.show()
