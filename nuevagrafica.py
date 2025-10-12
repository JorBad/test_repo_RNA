import numpy as np
import matplotlib.pyplot as plt

# ============================
# === 1. Datos del Sistema 1 ==
# ============================
errores_abs_1 = np.array([
    12.491638795986622, 11.155932203389831, 4.626262626262626, 2.0033783783783785,
    1.125, 0.6959459459459459, 0.959731543624161, 0.6354515050167224,
    0.7601351351351351, 1.239057239057239, 0.725752508361204
])

raices_1 = np.array([
    25.38695506430785, 24.353907705188103, 15.631544314781397,
    10.241223220945068, 7.424105800618382, 5.496752287877426,
    7.357398530140468, 5.18792568030052, 5.938889472221663,
    8.458918340567298, 4.991953324095664
])

errores_rel_1 = np.array([
    77.15897357484008, 64.97418717641808, 26.16951947476621, 12.163340192807885,
    6.356785688974804, 5.323016993282573, 4.079853113858246, 4.517189587353225,
    5.352089747815204, 5.7589146132415605, 5.58573302447802
])

# ============================
# === 2. Datos del Sistema 2 ==
# ============================
errores_abs_2 = np.array([
    13.956375838926174, 11.878787878787879, 6.837837837837838, 3.3557046979865772,
    2.5418060200668897, 2.802675585284281, 1.3945578231292517, 1.7591973244147157,
    2.016949152542373, 2.8277027027027026, 1.6195286195286196
])

raices_2 = np.array([
    27.71436195523647, 25.360747478935934, 19.756683188537618, 13.257359305592338,
    11.425968835030817, 12.709352390698543, 7.903411764913504, 9.483654279467505,
    10.361912505383877, 12.936067282396682, 8.764545560566658
])

errores_rel_2 = np.array([
    80.79139670095577, 67.33346826590338, 36.756171856447345, 19.58851855106026,
    17.088404119508564, 14.896672465329265, 8.594683879678314, 11.199768048637187,
    9.90169817082386, 12.562289321702691, 11.231428934930447
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
plt.plot(x, errores_abs_1, 'o-', label='Modelo de Fuchs', linewidth=2)
plt.plot(x, errores_abs_2, 's--', label='Mi modelo', linewidth=2)
plt.title('Comparación de Errores Absolutos')
plt.xlabel("SNR, [dB]")
plt.ylabel('Error absoluto [°]')
plt.legend()
plt.yscale("log")
plt.grid(True)

# --- Variable 2: Raíces ---
plt.subplot(3, 1, 2)
plt.plot(x, raices_1, 'o-', label='Modelo de Fuchs', linewidth=2)
plt.plot(x, raices_2, 's--', label='Mi modelo', linewidth=2)
plt.title('Comparación de Raíces')
plt.xlabel("SNR, [dB]")
plt.ylabel('RMSE')
plt.yscale("log")
plt.legend()
plt.grid(True)

# --- Variable 3: Errores relativos ---
plt.subplot(3, 1, 3)
plt.plot(x, errores_rel_1, 'o-', label='Modelo de Fuchs', linewidth=2)
plt.plot(x, errores_rel_2, 's--', label='Mi modelo', linewidth=2)
plt.title('Comparación de Errores Relativos (%)')
plt.xlabel("SNR, [dB]")
plt.ylabel('Error relativo (%)')
plt.legend()
plt.yscale("log")
plt.grid(True)

plt.tight_layout()
plt.show()
