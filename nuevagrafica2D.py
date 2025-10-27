import numpy as np
import matplotlib.pyplot as plt


## CODIGO QUE GENERA LA COMPARACIÓN ENTRE Numeros de antenas


# ============================
# === 1. Datos de 10x10 antenas ==
# ============================
errores_abs_1 = np.array([
    12.737373737373737, 12.757575757575758, 11.35, 10.72,
    4.111111111111111, 1.2371134020618557, 0.4375, 
    0.47959183673469385, 0.6391752577319587, 0.010309278350515464, 0.9081632653061225
])

raices_1 = np.array([
    29.85645932, 29.62314319, 27.66767066, 26.70580461, 16.14285714,  8.1449632,
  6.04691801,  6.78386566,  6.74773406,  0.14586499,  8.07129688
])

errores_rel_1 = np.array([
    96.54010615569557, 92.74877939819301, 92.54174959673081, 83.70063550909612, 
    38.830097317112774, 10.716585835922627, 2.2191489340555752, 2.0832890080352904, 
    4.570762882187371, 0.054553974884441046, 6.346556895404317
])

# ============================
# === 2. Datos del Sistema 2 ==
# ============================
errores_abs_2 = np.array([
    12.02, 11.747474747474747, 10.408163265306122, 11.377551020408163, 
    6.08, 3.68, 3.515463917525773, 2.02, 4.04, 0.3711340206185567, 5.783505154639175
])

raices_2 = np.array([
    28.588809,   27.38873595, 25.91492491, 27.90422608, 19.64993639, 15.22629305,
    16.46337004, 12.51718818, 17.34243351,  4.53356278, 21.7593218
])

errores_rel_2 = np.array([
    96.30175356267524, 99.99207061681825, 96.26292185313496, 92.96727464705377, 
    51.82768310819018, 27.103029640089098, 19.82319695587248, 12.125170166939233, 
    24.161169604149787, 2.434501132117013, 31.91395343856248
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
