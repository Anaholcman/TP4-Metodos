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

# Algoritmo de Gradiente Descendente

def gradient_descent(A, b, s, max_iter=1000):
    x = np.random.rand(A.shape[1])  # Condición inicial aleatoria
    errors = []
    norms = []  # Para registrar la norma 2 de x
    for _ in range(max_iter):
        error = cost_function(A, x, b)
        errors.append(error)
        norms.append(np.linalg.norm(x))  # Registrar la norma 2 de x
        grad = gradient(A, x, b)
        x = x - s * grad
    return x, errors, norms  # Devolver también la lista norms

def gradient_descent_regularized(A, b, s, delta, max_iter=1000):
    x = np.random.rand(A.shape[1])  # Condición inicial aleatoria
    errors = []
    norms = []  # Para registrar la norma 2 de x
    for _ in range(max_iter):
        error = cost_function_regularized(A, x, b, delta)
        errors.append(error)
        norms.append(np.linalg.norm(x))  # Registrar la norma 2 de x
        grad = gradient_regularized(A, x, b, delta)
        x = x - s * grad
    return x, errors, norms  # Devolver también la lista norms


# Generación de Datos Aleatorios
n, d = 5, 100
A = np.random.rand(n, d)
b = np.random.rand(n)

# Cálculo de H_F
H_F = 2 * np.dot(A.T, A)

# Cálculo de sigma_max y lambda_max
sigma_max = np.linalg.svd(A, compute_uv=False)[0]
lambda_max = np.linalg.eigvals(H_F).real.max()

# Parámetros
s = 1 / lambda_max
delta = 10**(-2) * sigma_max

# Ejecutar el algoritmo de gradiente descendente para F(x)
x_solution, errors, norms = gradient_descent(A, b, s)

# Ejecutar el algoritmo de gradiente descendente para F2(x) con regularización L2
x_solution_reg, errors_reg, norms_reg = gradient_descent_regularized(A, b, s, delta)

# Comparar con la solución obtenida mediante SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)
x_svd = np.dot(VT.T, np.dot(np.linalg.inv(np.diag(S)), np.dot(U.T, b)))
error_svd = np.linalg.norm(np.dot(A, x_svd) - b)

# Grafico la evolución del error
plt.plot(range(len(errors)), errors, label='F(x)', color='darkgreen')
plt.plot(range(len(errors_reg)), errors_reg, label='F2(x) con Regularización', color='purple')
plt.axhline(y=error_svd, color='orange', linestyle='--', label='SVD')
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.title('Evolución del Error')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

# Grafico la comparación de soluciones
plt.plot(x_solution, label='Gradiente Descendente F(x)')
plt.plot(x_solution_reg, label='Gradiente Descendente F2(x)', linestyle='dashed')
plt.plot(x_svd, label='SVD', linestyle='dotted', linewidth=1.5)
plt.xlabel('Índice de Componente')
plt.ylabel('Valor de Componente')
plt.title('Comparación de Soluciones')
plt.legend()
plt.show()

# Grafico la norma 2 de x en función de las iteraciones
plt.plot(range(len(norms)), norms, label='Norma 2 de x - F(x)', color='darkgreen')
plt.plot(range(len(norms_reg)), norms_reg, label='Norma 2 de x - F2(x)', color='purple')
plt.xlabel('Iteraciones')
plt.ylabel(r'$\|x\|_2$')
plt.title(r'$\|x\|_2$ en función de iteraciones')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

# Análisis del impacto de la regularización L2
delta_values = [10**(-2) * sigma_max, 10**(-1) * sigma_max, 10**(0) * sigma_max]

for delta in delta_values:
    x_solution_reg, errors_reg, norms_reg = gradient_descent_regularized(A, b, s, delta)
    plt.plot(range(len(errors_reg)), errors_reg, label=f'δ²={delta}')
    
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.title('Impacto de la Regularización L2')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()