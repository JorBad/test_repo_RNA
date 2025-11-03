import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import pickle

def datos1_2(angles_deg,array_pos,SNR_dB,K,num_samples,M,mapa,L,t,mapaphi,anglesphi):
  # Inicialización
  X_data = np.zeros((num_samples, 2*M*K))  # Entradas reales
  longi=len(mapa)+len(mapaphi)
  y_labels1 = np.zeros((num_samples,len(mapa)), dtype=int)  # Etiquetas binarias theta
  y_labels2 = np.zeros((num_samples,len(mapaphi)), dtype=int)  # Etiquetas binarias phi
  y_labels = np.empty((0, longi))
  srl=10**(SNR_dB/10);
  pruido=1/srl;
  # y_labels = np.zeros((num_samples,), dtype=int)  # Etiquetas de clase (índices)
  # plt.figure(figsize=(10, 6))
  #print(y_labels)
  thetas=np.zeros(L)
  phis=np.zeros(L)
  # Initialize steering_vector as a list to hold arrays
  steering_vectors = []
  steering_vectorsp = []
  for m in range(num_samples):
    stv=np.zeros((M,1))
    stvp=np.zeros((M,1))
    #print(stv)
    steering_vectors.clear() # Clear for each sample
    steering_vectorsp.clear() # Clear for each sample
    for i in range(L):
      # Ángulo verdadero (una clase aleatoria)
      thetas[i] = np.random.choice(angles_deg)
      theta_rad = np.deg2rad(thetas[i])
      phis[i] = np.random.choice(anglesphi)
      phi_rad = np.deg2rad(phis[i])
      # steering vector para lambda/4
      # stv = np.pi/2 * array_pos * np.sin(theta_rad)
      # steering vector para lambda/2
      # stv = np.pi * array_pos * np.sin(theta_rad)*np.sin(phi_rad) ## ORIGINAL
      stv = np.pi  * np.sin(theta_rad)*np.sin(phi_rad) ## DIAGONAL
      stvp = np.pi  * np.sin(theta_rad)*np.cos(phi_rad)
    #   print(stv.shape)
      # Append the steering vector to the list
      steering_vectors.append(stv.reshape(-1, 1))
      steering_vectorsp.append(stvp.reshape(-1, 1))
    print("thetas:",thetas)
    print("phis:",phis)
    print("Muestra:",m)
 
    R=np.zeros((M,K),complex)
    omeg= 2*np.pi*1e9;


    for n in range(K):#K
      tmp=np.zeros((M,1),complex) ##DIAGONAL
      #print(tmp)
    #   tmp=0
      for a in range(L):   
        for ant in range(M):    
            # indice= ant+((M-1)*(ant))
            # print("indice:",ant)
            # print("theta actual",thetas[a])
            # print("phi actual",phis[a])
            # tmp[indice:indice+(M)] = tmp[indice: indice+(M)]+ np.exp(1j*(omeg*t[n]-steering_vectors[a])) + np.exp(1j*(omeg*t[n]-steering_vectorsp[a]*array_pos[ant])) ## ORIGINAL
            tmp[ant] = tmp[ant]+ np.exp(1j*(omeg*t[n]-steering_vectors[a]*array_pos[ant])) + np.exp(1j*(omeg*t[n]-steering_vectorsp[a]*array_pos[ant])) ## DIAGONAL
            # print("Temporal",tmp)
      noise = (np.random.randn(M,1)) * pruido
      R[:,n] = (tmp + noise).flatten()
      # print(R)
    Z = np.concatenate([R.T.real.flatten(), R.T.imag.flatten()])

    X_data[m, :] = Z
  
    # y_labels[m] = mapa[thetas[0]]  # <----------------- PARA ETIQUETAS DE INDICES
    for theta in thetas:
     y_labels1[m, mapa[theta]] = 1  # etiqueta BINARIA de longitud G
    
    for phi in phis:
     y_labels2[m, mapaphi[phi]] = 1
    
    # y_labels = np.vstack((y_labels1, y_labels2))
    y_labels =np.concatenate((y_labels1, y_labels2), axis=1)
    # plt.plot(np.real(R[0,:]))
    # print(y_labels.shape)
    # plt.xlim(0,100)
  # plt.show
  return X_data, y_labels
def datos2_opti(angles_deg, array_pos, SNR_dB, K, num_samples, M, mapa, L, t, mapaphi, anglesphi):
    X_data = np.zeros((num_samples, 2 * M * M * K))
    longi = len(mapa) + len(mapaphi)
    y_labels1 = np.zeros((num_samples, len(mapa)), dtype=int)
    y_labels2 = np.zeros((num_samples, len(mapaphi)), dtype=int)
    srl = 10 ** (SNR_dB / 10)
    pruido = 1 / srl
    omeg = 2 * np.pi * 1e9

    array_pos = np.asarray(array_pos).reshape(-1, 1)  # Asegura forma columna

    for m in range(num_samples):
        thetas = np.random.choice(angles_deg, size=L)
        phis = np.random.choice(anglesphi, size=L)
        theta_rad = np.deg2rad(thetas).reshape(-1, 1)
        phi_rad = np.deg2rad(phis).reshape(-1, 1)
        print("thetas:",thetas)
        print("phis:",phis)
        print("Muestra:",m)

        # Vector de dirección para cada señal incidente
        sin_theta = np.sin(theta_rad)
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)

        # Vector de dirección para cada señal incidente
        steering_vectors = np.pi * array_pos @ (sin_theta.T * sin_phi.T)  # (M, L)
        steering_vectorsp = np.pi * (sin_theta * cos_phi).T  # (L,)

        R = np.zeros((M * M, K), dtype=complex)

        for n in range(K):
            tmp = np.zeros((M * M,), dtype=complex)
            for a in range(L):
                phase1 = np.exp(1j * (omeg * t[n] - steering_vectors[:, a]))
                phase2 = np.exp(1j * (omeg * t[n] - steering_vectorsp[a] * array_pos.flatten()))
                for ant in range(M):
                    idx = ant * M 
                    if idx + M <= M * M:
                        tmp[idx:idx + M] += phase1.flatten() + phase2
            noise = np.random.randn(M * M) * pruido
            R[:, n] = tmp + noise

        Z = np.concatenate([R.T.real.flatten(), R.T.imag.flatten()])
        X_data[m, :] = Z

        # Etiquetado binario
        y_labels1[m, [mapa[theta] for theta in thetas]] = 1
        y_labels2[m, [mapaphi[phi] for phi in phis]] = 1

    y_labels = np.concatenate((y_labels1, y_labels2), axis=1)
    return X_data, y_labels

from typing import List, Tuple, Union
Number = Union[int, float]
def emparejar(lista1: List[Number],
                             lista2: List[Number],
                             tol: Number = 5,
                             fill: Number = 0) -> List[Tuple[Number, Number]]:
    # Normaliza tipos
    l1 = [float(x) for x in lista1]
    l2 = [float(x) for x in lista2]

    usados2 = [False]*len(l2)
    pares: List[Tuple[Number, Number]] = []

    # 1) Recorre lista1 (predicciones) y busca su etiqueta real más cercana
    for pred in l1:
        candidato_idx = -1
        mejor_diff = float('inf')

        for j, real in enumerate(l2):
            if usados2[j]:
                continue
            diff = abs(pred - real)
            if diff <= tol and diff < mejor_diff:
                mejor_diff = diff
                candidato_idx = j

        if candidato_idx >= 0:
            pares.append((int(round(l2[candidato_idx])), int(round(pred))))
            usados2[candidato_idx] = True
        else:
            pares.append((fill, int(round(pred))))  # no hay real → 0

    # 2) Agrega reales que quedaron sin predicción
    for j, real in enumerate(l2):
        if not usados2[j]:
            pares.append((int(round(real)), fill))

    return pares

def dataset_(X_data, y_labels):
  #X_data, y_labels = crearsignals(M,K,SNR_dB,precision,num_samples)


  # Separar conjunto de entrenamiento y validación
  X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

  # Dataset personalizado
  class DOADataset(Dataset):
      def __init__(self, X, y):
          self.X = torch.tensor(X, dtype=torch.float32)
          self.y = torch.tensor(y, dtype=torch.long)

      def __len__(self):
          return len(self.y)

      def __getitem__(self, idx):
          return self.X[idx], self.y[idx]

  # Instancias
  train_dataset = DOADataset(X_train, y_train)
  val_dataset = DOADataset(X_val, y_val)

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
  #print(f"-----------Se separó el dataset -------")
  return train_loader, val_loader, X_train, X_val, y_train, y_val


def modelo(M,K,nclases):
  #clases= 180/precision
  clases=nclases
  class DeepMLP(nn.Module):
      def __init__(self, input_size=2*M*M*K, output_size=clases):
          super(DeepMLP, self).__init__()
          self.model = nn.Sequential(
              nn.Linear(input_size, 512),
              # nn.ReLU(),
              nn.Tanh(),
              nn.Dropout(0.3),
              nn.Linear(512, 256),
              # nn.ReLU(),
              nn.Tanh(),
              nn.Dropout(0.3),
              nn.Linear(256, 128),
              nn.ReLU(),
              # nn.Tanh(),
              nn.Linear(128, clases)
          )

      def forward(self, x):
          return self.model(x)


  # Instancia del modelo
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #model = MLPClassifier().to(device)
  model = DeepMLP().to(device)
  return model,device

def modelo3(M,K,clases):
  #clases= 180/precision
  class DeepMLP(nn.Module):
      def __init__(self, input_size=2*M*K, output_size=clases):
          super(DeepMLP, self).__init__()
          self.model = nn.Sequential(
              nn.Linear(input_size, 1024),
              nn.ReLU(),
              # nn.Tanh(),
              # nn.Sigmoid(),
              nn.Dropout(0.3),
              nn.Linear(1024, 1024),
              nn.ReLU(),
              # nn.Sigmoid(),
              # nn.Tanh(),
              nn.Dropout(0.3),
              nn.Linear(1024, 1024),
              nn.ReLU(),
              # nn.Sigmoid(),
              # nn.Tanh(),
              nn.Dropout(0.3),
              nn.Linear(1024, 1024),
              nn.ReLU(),
              # nn.Sigmoid(),
              # nn.Tanh(),
              nn.Linear(1024, int(clases))
          )

      def forward(self, x):
          return self.model(x)

  # Instancia del modelo
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #model = MLPClassifier().to(device)
  model = DeepMLP().to(device)
  return model,device

M=10
K=1
num_samples=1000
SNR_dB=0.1;
precision=5
lr=0.0001
fs = 10e9;
L=1
# t = (0:1/fs:1e-6);
t= np.arange(0,1e-6,1/fs);

angles_de2 = np.arange(0, 90, precision)
angles_de = np.arange(0, 90, precision) ### <------- para datos de prueba
angles_phi = np.arange(0, 360, 180)
# clases= len(angles_de2)
clases= len(angles_de)+len(angles_phi)
# print(angles_de)
model,device=modelo3(M,K,clases) ###### <-------- MODELO
# 2. Carga los pesos
model.load_state_dict(torch.load("modelo1_2Dp1_100.pth"))

array_po = np.linspace(0, (M-1), M)
#angles_de = np.arange(0, 90, 5)
mapas = {valor: idx for idx, valor in enumerate(angles_de)}
mapas2 = {valor: idx for idx, valor in enumerate(angles_de2)}
n_total=num_samples
numdatos= len(angles_de)
base_count = n_total // numdatos  #
restantes = n_total % numdatos    #

# Inicializar la lista de salida
result = []

  # Añadir base_count veces cada elemento
for angle in angles_de:
  result.extend([angle] * base_count)

for i in range(restantes):
  result.append(angles_de[i])

  # Convertir a array si se desea
result = np.array(result)

np.random.shuffle(result)
# print(result)
mapas_phi = {valor: idx for idx, valor in enumerate(angles_phi)}
#   print("mapas de phi:",mapas_phi)
#   n_total=num_samples
numdatos_phi= len(angles_phi)
base_count_phi = n_total // numdatos_phi  #
restantes_p = n_total % numdatos_phi    #

  # Inicializar la lista de salida
result_p = []

  # Añadir base_count veces cada elemento
for angle in angles_phi:
   result_p.extend([angle] * base_count_phi)

  # Añadir los elementos restantes (para completar los 100)
  # Puedes seleccionar los primeros "restantes" elementos de x
for i in range(restantes_p):
  result_p.append(angles_phi[i])

  # Convertir a array si se desea
result_p = np.array(result_p)
np.random.shuffle(result_p)


longi=len(mapas)+len(mapas_phi)

ruidos=[-20,-15,-10,-5,0,5,10,15,20,25,30]
ruidos1 =[5,10]
Pglobal=[]
absGlobal=[]
raizGlobal=[]
pPglobal=[]
pabsGlobal=[]
praizGlobal=[]
for ss in ruidos:
    X_datax, y_labels = datos1_2(result,array_po,SNR_dB=ss,K=K,num_samples=num_samples,M=M,mapa=mapas,L=L,t=t,mapaphi=mapas_phi,anglesphi=result_p)
    train_loader, val_loader, X_train, X_val, y_train, y_val=dataset_(X_datax, y_labels)
    inp=torch.tensor(X_datax, dtype=torch.float32).to(device)
    # print()

    
    with torch.no_grad():
      output = model(inp)
      # prediction = torch.argmax(output, dim=1)
      prediction = torch.argmax(output, dim=1).cpu().numpy()
      predicted = (output > 0.5).int().cpu().numpy()  # Convierte a 0 o 1

    # print("prediccion:",predicted[0])
    er = []
    erabs=[]
    ermse=[]
    per = []
    perabs=[]
    permse=[]
    mapa_inv = {v: k for k, v in mapas.items()}  # inverso: índice → theta

    mapa_inv2 = {v: k for k, v in mapas2.items()}  # inverso: índice → theta
    mapa_invp = {v: k for k, v in mapas_phi.items()}  # mapa inverso: índice → phi
    for k in range(len(predicted)):
        # Encuentra los índices donde hay 1
        indices_p = np.where(predicted[k] == 1)[0]
        indices_y = np.where(y_labels[k] == 1)[0]
        # Convierte a los valores reales de theta
        xy=0
        indicestheta=[]
        indicesphi=[]
        for xy in range(len(indices_y)):
              # print(xy)
            if indices_y[xy] < 90/precision:
                indicestheta.append(indices_y[xy])
            else: indicesphi.append(indices_y[xy]-90/precision)

        thetas_reales = [mapa_inv[i] for i in indicestheta]
        phis_reales = [mapa_invp[i] for i in indicesphi]
        # print("phis reales",phis_reales)
        indicestheta=[]
        indicesphi=[]
        xy=0
        for xy in range(len(indices_p)):
          #print(xy)
            if indices_p[xy] < 90/precision:
                indicestheta.append(indices_p[xy])
            else: indicesphi.append(indices_p[xy]-90/precision)
        thetas_p = [mapa_inv[i] for i in indicestheta]
            # print("indices phi:",indicesphi)
        phis_p = [mapa_invp[i] for i in indicesphi]
        # print("thetas predichas:", thetas_p)
      
        print(f"predicha: {phis_p}")
        # print(f"real: {phis_reales}")
        xx= emparejar(thetas_p, thetas_reales)
        pp= emparejar(phis_p, phis_reales)
        print("Parejas:",pp)   
        er2=[]
        erabs2=[]
        ### THETAS
        for s in range(len(xx)):
          if xx[s][0]!=0:
            fallo= float(abs((xx[s][0]-xx[s][1])/(xx[s][0]+1e-3))*100)
            raiz= float((xx[s][1]-xx[s][0])**2)
            falloabs=float(abs(xx[s][1]-xx[s][0]))
            # print("raiz:",raiz)
            er.append(fallo)
            ermse.append(raiz)
            erabs.append(falloabs)
        ## PHIS
        for s in range(len(pp)):
          if pp[s][0]!=0:
            fallop= float(abs((pp[s][0]-pp[s][1])/(pp[s][0]+1e-3))*100)
            raizp= float((pp[s][1]-pp[s][0])**2)
            falloabsp=float(abs(pp[s][1]-pp[s][0]))
            print("raiz phi:",fallop)
            per.append(fallop)
            permse.append(raizp)
            perabs.append(falloabsp)
        

    ############### THETAS ###########
    media = sum(er) / len(er)
    # print('RMSE',er)
    Pglobal.append(media)

    raices= np.sqrt(sum(ermse) / len(ermse))
    raizGlobal.append(raices)

    mediaabs= sum(erabs) / len(erabs)
    absGlobal.append(mediaabs)

    ############### phis ###########
    pmedia = sum(per) / len(per)
    # print('RMSE',er)
    pPglobal.append(pmedia)

    praices= np.sqrt(sum(permse) / len(permse))
    praizGlobal.append(praices)

    pmediaabs= sum(perabs) / len(perabs)
    pabsGlobal.append(pmediaabs)
    
    # print("RMSE de phi:",praices)

raizGlobal=np.array(raizGlobal).astype(float)
praizGlobal=np.array(praizGlobal).astype(float)
# mmedia= sum(er) / len(er)
# print("Promedio de RMSE:",mmedia)
# Nabs = sum(absGlobal)/len(absGlobal)
# print("Promedio de error absoluto:",Nabs)
# print(len(absGlobal))
# ruidos2=[-20,-15,-10,-5,0,5,10,15,20,25,30]
plt.plot(ruidos,raizGlobal)
plt.yscale("log")
plt.ylabel("RMSE")
plt.xlabel("SNR, [dB]")
print("ErroresAbs:",absGlobal)
print("Raices:",raizGlobal)
print("ErroresRelativos:",Pglobal)

print("ErroresAbs phi:",pabsGlobal)
print("Raices phi:",praizGlobal)
print("ErroresRelativos phi:",pPglobal)
plt.show();

