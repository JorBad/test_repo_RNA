import numpy as np
import matplotlib.pyplot as plt


## CODIGO QUE GENERA LA COMPARACIÓN ENTRE datos de entrenamiento



# ============================
# === 1. 200 k ==
# ============================
errores_abs_1 = np.array([
    47.36, 50.60606060606061, 47.81, 51.464646464646464, 
    55.76, 43.07070707070707, 46.08, 48.121212121212125, 40.25, 
    48.144329896907216, 41.55
])

raices_1 = np.array([
    55.68015805, 58.20548761, 55.37391082, 59.1639564,  62.41586337, 53.07170678,      
 55.39530666, 57.66097362, 51.1645385,  57.58740555, 50.84063336
])

errores_rel_1 = np.array([
    99.99454643125591, 99.99388113713613, 99.06613518133547, 99.06759163605356, 
    98.02757859291752, 88.88321083420739, 91.01674494420463, 90.93801440589256, 
    87.02703917569033, 93.80944794127883, 90.15209455044342
])

# ============================
# === 2. Datos 7*7 ==
# ============================
errores_abs_2 = np.array([
    53.888888888888886, 48.84, 47.07142857142857, 50.833333333333336, 
    52.84848484848485, 44.608247422680414, 49.6, 42.98, 50.704081632653065, 
    43.13131313131313, 49.3
])

raices_2 = np.array([
    60.39006206, 56.57437583, 56.68378093, 58.45119046, 59.57662073, 53.94508472,      
 56.87477472, 53.53148606, 60.20483742, 53.59462509, 58.76512571
])

errores_rel_2 = np.array([
    99.99644603917237, 98.19939515754528, 98.97204924620472, 97.95359516392917, 
    98.02018815561931, 90.77171494955434, 96.99598482155163, 86.01604738094714, 
    94.9877147113073, 90.90436098807758, 92.01813601717016
])

# ============================
# === Datos del modelo para 200k entradas ==
# ============================
errores_abs_3 = np.array([
    20.86, 22.88, 24.714285714285715, 24.612244897959183, 19.32, 
    18.06122448979592, 17.142857142857142, 15.54, 13.9, 15.326530612244898, 17.52
])

raices_3 = np.array([
    27.91239941, 28.31739748, 27.74361788, 28.01384497, 28.58283967, 25.47587094,
 23.30064377, 24.32652873, 21.8878962,  24.46303334, 22.82542442
])

errores_rel_3 = np.array([
    98.15134777359691, 98.46379803476266, 98.14758154173579, 98.63362516276216, 
    81.18858049100635, 58.67459128617672, 40.52067864177295, 49.3239371791868, 
    46.48375822823746, 52.069619919232444, 49.34009810292606
])
# ==========================================
# === 3. Eje X (definido por el usuario) ===
# ==========================================
# x = np.arange(1, 12)  # Por ejemplo, 1 a 11
x=[-20,-15,-10,-5,0,5,10,15,20,25,30]

# ==========================================
# === 4. Graficar comparaciones ============
# ==========================================

plt.figure(figsize=(12, 10))

# --- Variable 1: Errores absolutos ---
plt.subplot(3, 1, 1)
plt.plot(x, errores_abs_1, 'o-', label='10 antenas por lado', linewidth=2)
plt.plot(x, errores_abs_2, 's--', label='7 antenas por lado', linewidth=2)
plt.plot(x, errores_abs_3, 's-', label='5 antenas por lado', linewidth=2)
plt.title('Comparación de Errores Absolutos')
plt.xlabel("SNR, [dB]")
plt.ylabel('Error absoluto [°]')
plt.legend()
plt.yscale("log")
plt.grid(True)

# --- Variable 2: Raíces ---
plt.subplot(3, 1, 2)
plt.plot(x, raices_1, 'o-', label='10 antenas por lado', linewidth=2)
plt.plot(x, raices_2, 's--', label='7 antenas por lado', linewidth=2)
plt.plot(x, raices_3, 's-', label='5 antenas por lado', linewidth=2)
plt.title('Comparación de Raíces')
plt.xlabel("SNR, [dB]")
plt.ylabel('RMSE')
plt.yscale("log")
plt.legend()
plt.grid(True)

# --- Variable 3: Errores relativos ---
plt.subplot(3, 1, 3)
plt.plot(x, errores_rel_1, 'o-', label='10 antenas por lado', linewidth=2)
plt.plot(x, errores_rel_2, 's--', label='7 antenas por lado', linewidth=2)
plt.plot(x, errores_rel_3, 's-', label='5 antenas por lado', linewidth=2)
plt.title('Comparación de Errores Relativos (%)')
plt.xlabel("SNR, [dB]")
plt.ylabel('Error relativo (%)')
plt.legend()
plt.yscale("log")
plt.grid(True)

plt.tight_layout()
plt.show()
