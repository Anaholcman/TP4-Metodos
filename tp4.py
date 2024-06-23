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

def generate_bad_conditioned_matrix(A):    
    # Realizar una descomposición en valores singulares (SVD)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Forzar a que los valores singulares sean razonables para tener una matriz bien condicionada
    # Por ejemplo, establecer los valores singulares entre 1 y 10
    s = np.linspace(1, 10, n)
    
    # Reconstruir la matriz con los valores singulares ajustados
    A_well_conditioned = np.dot(U * s, Vt)
    
    return A_well_conditioned

def calcular_numero_condicion(A):
    # Calcular el número de condición usando la norma 2
    numero_condicion = np.linalg.cond(A, p=2)
    
    return numero_condicion

def inicializar_parametros(A, b):

    H_F = 2 * np.dot(A.T, A)
    
    sigma_max = np.linalg.svd(A, compute_uv=False)[0]
    lambda_max = np.linalg.eigvals(H_F).real.max()

    s = 1 / lambda_max
    delta = 10**(-2) * sigma_max

    x_solution, errors, norms = gradient_descent(A, b, s)

    x_solution_reg, errors_reg, norms_reg = gradient_descent_regularized(A, b, s, delta)

    U, S, VT = np.linalg.svd(A, full_matrices=False)
    x_svd = VT.T @ np.diag(1 / S) @ U.T @ b
    norm_svd = np.linalg.norm(np.dot(A, x_svd) - b, ord=2)
    error_svd = cost_function(A, x_svd, b)
    
    return s, delta, x_solution, errors, norms, x_solution_reg, errors_reg, norms_reg, x_svd, norm_svd, error_svd

# Generación de Datos Aleatorios
n, d = 5, 100
A = np.random.randn(n, d)
b = np.random.randn(n)

A_moño = generate_bad_conditioned_matrix(A)

s, delta, x_solution, errors, norms, x_solution_reg, errors_reg, norms_reg, x_svd, norm_svd, error_svd = inicializar_parametros(A, b)
s_moño, delta_moño, x_solution_moño, errors_moño, norms_moño, x_solution_reg_moño, errors_reg_moño, norms_reg_moño, x_svd_moño, norm_svd_moño, error_svd_moño = inicializar_parametros(A_moño, b)

a = False
errors2 = []
for elem in errors:
    if elem <= error_svd:
        a = True
    if a:
        errors2.append(error_svd)
    else:
        errors2.append(elem)
    

# Grafico la evolución del error
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(errors2)), errors2, label='F(x)', color='darkgreen')
plt.plot(range(len(errors_reg)), errors_reg, label='F2(x) con Regularización', color='purple')
plt.axhline(y=error_svd, color='orange', linestyle='--', label='SVD')
plt.xlabel('Iteraciones', fontsize=16)
plt.ylabel('Costo', fontsize=16)
plt.title(f'Número de Condición: {calcular_numero_condicion(A)}', fontsize=18)
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.plot(range(len(errors_moño)), errors_moño, label='F(x)', color='darkgreen')
plt.plot(range(len(errors_reg_moño)), errors_reg_moño, label='F2(x) con Regularización', color='purple')
plt.axhline(y=error_svd_moño, color='orange', linestyle='--', label='SVD')
plt.xlabel('Iteraciones', fontsize=16)
plt.ylabel('Costo', fontsize=16)
plt.title(f'Número de Condición: {calcular_numero_condicion(A_moño)}', fontsize=18)
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.suptitle('Evolución del Error', fontsize=19)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# # Grafico la comparación de soluciones
# plt.figure()  
# plt.subplot(1, 2, 1)
# plt.plot(x_solution, label='Gradiente Descendente F(x)')
# plt.plot(x_solution_reg, label='Gradiente Descendente F2(x)', linestyle='dashed')
# plt.plot(x_svd, label='SVD', linestyle='dotted', linewidth=1.5)
# plt.xlabel('Índice de Componente', fontsize=15)
# plt.ylabel('Valor de Componente', fontsize=15)
# plt.legend()
# plt.title('Matriz Random')

# plt.subplot(1, 2, 2)
# plt.plot(x_solution_moño, label='Gradiente Descendente F(x)')
# plt.plot(x_solution_reg_moño, label='Gradiente Descendente F2(x)', linestyle='dashed')
# plt.plot(x_svd_moño, label='SVD', linestyle='dotted', linewidth=1.5)
# plt.xlabel('Índice de Componente', fontsize=15)
# plt.ylabel('Valor de Componente', fontsize=15)
# plt.legend()
# plt.title('Matriz Bien Condicionada')

# plt.suptitle('Comparación de Soluciones')
# plt.tight_layout()
# plt.show()

# Grafico la norma 2 de x en función de las iteraciones
plt.figure()  
plt.plot(range(len(norms)), norms, label='Norma 2 de x - F(x)', color='darkgreen')
plt.plot(range(len(norms_reg)), norms_reg, label='Norma 2 de x - F2(x)', color='purple')
plt.axhline(y=norm_svd, color='orange', linestyle='--', label='SVD')
plt.xlabel('Iteraciones')
plt.ylabel(r'$\|x\|_2$')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.title(r'$\|x\|_2$ en función de iteraciones')
plt.show()

# Análisis del impacto de la regularización L2
delta_values = [10**(-2), 10**(-1), 10**(0), 10]
plt.figure() 

for delta in delta_values:
    x_solution_reg, errors_reg, norms_reg = gradient_descent_regularized(A, b, s, delta)
    plt.plot(range(len(errors_reg)), errors_reg, label=f'δ²={delta}')
    
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.title('Impacto de la Regularización L2')
plt.show()


# Numero de condicion vs iteraciones
# Parámetros iniciales

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

# Generar matrices condicionadas
def generate_conditioned_matrix(n, d, condition_number):
    A = np.random.randn(n, d)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s = np.linspace(1, condition_number, min(n, d))
    A_conditioned = U @ np.diag(s) @ Vt
    return A_conditioned

# Inicializar parámetros para matrices condicionadas
def inicializar_parametros_condicionadas(A, b):
    H_F = 2 * np.dot(A.T, A)
    lambda_max = np.linalg.eigvals(H_F).real.max()
    s = 1 / lambda_max
    delta = 10**(-2) * np.linalg.norm(A, 2)
    x_solution, errors, norms = gradient_descent(A, b, s)
    return s, delta, x_solution, errors, norms

# Generación de Datos Aleatorios
n, d = 5, 100
threshold = 1e-8
condition_numbers = np.linspace(1, 200, 200)
iterations_needed = []

for condition_number in condition_numbers:
    A_conditioned = generate_conditioned_matrix(n, d, condition_number)
    b_conditioned = np.random.randn(n)
    s, delta, x_solution, errors, norms = inicializar_parametros_condicionadas(A_conditioned, b_conditioned)
    iterations = 0
    for error in errors:
        if error < threshold:
            break
        iterations += 1
    iterations_needed.append(iterations)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.semilogy(condition_numbers, iterations_needed, label='Iteraciones Numéricas', color='purple')
plt.semilogy(condition_numbers, condition_numbers**2, 'r--', label='Cota Teórica', color='orange')
plt.xlabel('Número de Condición κ(A)')
plt.ylabel('Número de Iteraciones')
plt.title('Convergencia del Gradiente Descendente en Función del Número de Condición')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.xlim([0, 35])
plt.ylim([1, 1e4])
plt.show()