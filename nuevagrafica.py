import numpy as np
import matplotlib.pyplot as plt


## CODIGO QUE GENERA LA COMPARACIÓN ENTRE FUCHS Y NOSOTROS


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
    12.506711409395972, 10.308474576271186, 3.249158249158249, 0.7635135135135135, 
    0.6148648648648649, 0.8993288590604027, 0.7643097643097643, 0.8885135135135135, 
    1.121212121212121, 0.5933333333333334, 0.6778523489932886
])

raices_2 = np.array([
    26.16207334, 22.28026608, 13.16798918,  8.78054316,  6.20693497,  4.47326514,
  5.98658467,  5.051991, 5.61612111,  4.84452142,  7.47014259
])

errores_rel_2 = np.array([
    73.62158254842291, 59.73281541039179, 30.155237628245253, 9.752324534496399, 
    6.0738527447634345, 4.420868425809209, 6.195761499101364, 8.410425297979653, 
    4.300540205362438, 6.480196656778979, 7.265170181448091
])

# ============================
# === Datos del modelo para 200k entradas ==
# ============================
errores_abs_3 = np.array([
    13.53924914675768, 10.01010101010101, 3.2576271186440677, 0.7190635451505016,
      1.227891156462585, 0.5709459459459459, 0.6262626262626263, 0.8355704697986577, 
      0.5661016949152542, 0.9264214046822743, 0.4478114478114478
])

raices_3 = np.array([
    27.04946534, 22.64950331, 13.02049666,  5.4877662,   8.36259467,  3.73961146,
  5.11690738,  6.41140083,  4.26374098,  6.81669544,  4.05106992
])

errores_rel_3 = np.array([
    78.02409843307969, 59.46466039536174, 23.074006881939237, 4.15540603145154, 
    8.905774859595255, 4.970423730865655, 5.117818903570975, 5.166214241743635, 
    3.6429691999065783, 5.119012387028041, 3.290345219882142
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
plt.plot(x, errores_abs_1, 'o-', label='Modelo de Fuchs 40k datos', linewidth=2)
plt.plot(x, errores_abs_2, 's--', label='Modelo de Fuchs 80k datos', linewidth=2)
plt.plot(x, errores_abs_3, 's-', label='Modelo de Fuchs 200k datos', linewidth=2)
plt.title('Comparación de Errores Absolutos')
plt.xlabel("SNR, [dB]")
plt.ylabel('Error absoluto [°]')
plt.legend()
plt.yscale("log")
plt.grid(True)

# --- Variable 2: Raíces ---
plt.subplot(3, 1, 2)
plt.plot(x, raices_1, 'o-', label='Modelo de Fuchs 40k datos', linewidth=2)
plt.plot(x, raices_2, 's--', label='Modelo de Fuchs 80k datos', linewidth=2)
plt.plot(x, raices_3, 's-', label='Modelo de Fuchs 200k datos', linewidth=2)
plt.title('Comparación de Raíces')
plt.xlabel("SNR, [dB]")
plt.ylabel('RMSE')
plt.yscale("log")
plt.legend()
plt.grid(True)

# --- Variable 3: Errores relativos ---
plt.subplot(3, 1, 3)
plt.plot(x, errores_rel_1, 'o-', label='Modelo de Fuchs 40k datos', linewidth=2)
plt.plot(x, errores_rel_2, 's--', label='Modelo de Fuchs 80k datos', linewidth=2)
plt.plot(x, errores_rel_3, 's-', label='Modelo de Fuchs 200k datos', linewidth=2)
plt.title('Comparación de Errores Relativos (%)')
plt.xlabel("SNR, [dB]")
plt.ylabel('Error relativo (%)')
plt.legend()
plt.yscale("log")
plt.grid(True)

plt.tight_layout()
plt.show()
