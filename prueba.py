import numpy as np
import pickle

# with open("output.pkl", "rb") as f:
#     output = pickle.load(f)

# print(output[2]
L=3
alpha=np.ones(4)
print(alpha)

import matplotlib.pyplot as plt

# # Datos del primer conjunto 
x1 = [0.01, 1, 10, 100]
y1 = [49.23, 91.23, 99.99, 99.99]

# # Datos del segundo conjunto
x2 = [0.01, 0.01,1, 10, 100]
y2 = [18.47,19.11,65.21,99.99,99.99]

# # Datos del tercer conjunto
x3 = [0.01, 0.01,1, 10, 100]
y3 = [8.63,13.64,51.11,99.99,99.99]

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, marker='o', linestyle='-', color='blue', label=r'10 épocas')
plt.plot(x2, y2, marker='s', linestyle='-', color='red', label=r'40 épocas')
plt.plot(x3, y3, marker='p', linestyle='-', color='purple', label=r'40 épocas y 500k')

# # Etiquetas y título
plt.xlabel('Ruido')
plt.ylabel('Error relativo')
# plt.title('Comparación')
plt.grid(True)
plt.legend()

# Mostrar la gráfica
plt.xscale("log")
plt.show()

# import matplotlib.pyplot as plt

# # Datos de ejemplo
# categorias = ["A", "B", "C", "D", "E"]
# valores = [23, 17, 35, 29, 12]

# # Crear gráfica de barras
# plt.bar(categorias, valores)

# # Añadir título y etiquetas
# plt.title("Ejemplo de gráfica de barras")
# plt.xlabel("Categorías")
# plt.ylabel("Valores")

# # Mostrar gráfica
# plt.show()
