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

def generate_well_conditioned_matrix(A):    
    # Realizar una descomposición en valores singulares (SVD)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Forzar a que los valores singulares sean razonables para tener una matriz bien condicionada
    # Por ejemplo, establecer los valores singulares entre 1 y 10
    s = np.linspace(1, 10, n)
    
    # Reconstruir la matriz con los valores singulares ajustados
    A_well_conditioned = np.dot(U * s, Vt)
    
    return A_well_conditioned

# Generación de Datos Aleatorios
n, d = 5, 100
A = np.random.rand(n, d)
b = np.random.rand(n)

# generar una matriz bien condicionada
A_moño = generate_well_conditioned_matrix(A)

# Cálculo de H_F
H_F = 2 * np.dot(A.T, A)

H_F_moño = 2 * np.dot(A_moño.T, A_moño)

# Cálculo de sigma_max y lambda_max
sigma_max = np.linalg.svd(A, compute_uv=False)[0]
lambda_max = np.linalg.eigvals(H_F).real.max()

sigma_max_moño = np.linalg.svd(A_moño, compute_uv=False)[0]
lambda_max_moño = np.linalg.eigvals(H_F_moño).real.max()

# Parámetros
s = 1 / lambda_max
delta = 10**(-2) * sigma_max

s_moño = 1 / lambda_max_moño
delta_moño = 10**(-2) * sigma_max_moño

# Ejecutar el algoritmo de gradiente descendente para F(x)
x_solution, errors, norms = gradient_descent(A, b, s)

x_solution_moño, errors_moño, norms_moño = gradient_descent(A_moño, b, s_moño)

# Ejecutar el algoritmo de gradiente descendente para F2(x) con regularización L2
x_solution_reg, errors_reg, norms_reg = gradient_descent_regularized(A, b, s, delta)

x_solution_reg_moño, errors_reg_moño, norms_reg_moño = gradient_descent_regularized(A_moño, b, s_moño, delta_moño)

# Comparar con la solución obtenida mediante SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)
x_svd = np.dot(VT.T, np.dot(np.linalg.inv(np.diag(S)), np.dot(U.T, b)))
error_svd = np.linalg.norm(np.dot(A, x_svd) - b)

U_moño, S_moño, VT_moño = np.linalg.svd(A_moño, full_matrices=False)
x_svd_moño = np.dot(VT_moño.T, np.dot(np.linalg.inv(np.diag(S_moño)), np.dot(U_moño.T, b)))
error_svd_moño = np.linalg.norm(np.dot(A_moño, x_svd_moño) - b)

# Grafico la evolución del error
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(range(len(errors)), errors, label='F(x)', color='darkgreen')
plt.plot(range(len(errors_reg)), errors_reg, label='F2(x) con Regularización', color='purple')
plt.axhline(y=error_svd, color='orange', linestyle='--', label='SVD')
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.title('Matriz Random')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.subplot(2, 2, 1)
plt.plot(range(len(errors_moño)), errors_moño, label='F(x)', color='darkgreen')
plt.plot(range(len(errors_reg_moño)), errors_reg_moño, label='F2(x) con Regularización', color='purple')
plt.axhline(y=error_svd_moño, color='orange', linestyle='--', label='SVD')
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.title('Matriz Bien Condicionada')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.suptitle('Evolución del Error')
plt.tight_layout()
plt.show()

# Grafico la comparación de soluciones
plt.figure()  
plt.subplot(2, 1, 1)
plt.plot(x_solution, label='Gradiente Descendente F(x)')
plt.plot(x_solution_reg, label='Gradiente Descendente F2(x)', linestyle='dashed')
plt.plot(x_svd, label='SVD', linestyle='dotted', linewidth=1.5)
plt.xlabel('Índice de Componente')
plt.ylabel('Valor de Componente')
plt.legend()
plt.title('Matriz Random')

plt.subplot(2, 2, 1)
plt.plot(x_solution_moño, label='Gradiente Descendente F(x)')
plt.plot(x_solution_reg_moño, label='Gradiente Descendente F2(x)', linestyle='dashed')
plt.plot(x_svd_moño, label='SVD', linestyle='dotted', linewidth=1.5)
plt.xlabel('Índice de Componente')
plt.ylabel('Valor de Componente')
plt.legend()
plt.title('Matriz Bien Condicionada')

plt.suptitle('Comparación de Soluciones')
plt.tight_layout()
plt.show()

# Grafico la norma 2 de x en función de las iteraciones
plt.figure()  
plt.subplot(2, 1, 1)
plt.plot(range(len(norms)), norms, label='Norma 2 de x - F(x)', color='darkgreen')
plt.plot(range(len(norms_reg)), norms_reg, label='Norma 2 de x - F2(x)', color='purple')
plt.xlabel('Iteraciones')
plt.ylabel(r'$\|x\|_2$')
plt.title('Matriz Random')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.subplot(2, 2, 1)
plt.plot(range(len(norms_moño)), norms_moño, label='Norma 2 de x - F(x)', color='darkgreen')
plt.plot(range(len(norms_reg_moño)), norms_reg_moño, label='Norma 2 de x - F2(x)', color='purple')
plt.xlabel('Iteraciones')
plt.ylabel(r'$\|x\|_2$')
plt.title('Matriz Bien Condicionada')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.suptitle(r'$\|x\|_2$ en función de iteraciones')
plt.tight_layout()
plt.show()

# Análisis del impacto de la regularización L2
# delta_values = [10**(-2) * sigma_max, 10**(-1) * sigma_max, 10**(0) * sigma_max]
delta_values = [10**(-3), 10**(-2), 10**(-1), 10**(0), 10]
plt.figure() 
plt.subplot(2, 1, 1)

for delta in delta_values:
    x_solution_reg, errors_reg, norms_reg = gradient_descent_regularized(A, b, s, delta)
    plt.plot(range(len(errors_reg)), errors_reg, label=f'δ²={delta}')
    
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.title('Matriz Random')

plt.subplot(2, 2, 1)

for delta in delta_values:
    x_solution_reg, errors_reg, norms_reg = gradient_descent_regularized(A_moño, b, s_moño, delta)
    plt.plot(range(len(errors_reg)), errors_reg, label=f'δ²={delta}')

plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.title('Matriz Bien Condicionada')

plt.suptitle('Impacto de la Regularización L2')
plt.tight_layout()
plt.show()
