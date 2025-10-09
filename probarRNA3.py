import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import pickle

def datos1_2(angles_deg,array_pos,SNR_dB,K,num_samples,M,mapa,L,t):
  # Inicialización
  X_data = np.zeros((num_samples, 2*M*K))  # Entradas reales
  
  y_labels = np.zeros((num_samples,len(mapa)), dtype=int)  # Etiquetas de clase binarias
  srl=10**(SNR_dB/10);
  pruido=1/srl;
  #print(y_labels)
  thetas=np.zeros(L)
  # Initialize steering_vector as a list to hold arrays
  steering_vectors = []
  for m in range(num_samples):
    stv=np.zeros((M,1))
    #print(stv)
    steering_vectors.clear() # Clear for each sample
    for i in range(L):
      # Ángulo verdadero (una clase aleatoria)
      thetas[i] = np.random.choice(angles_deg)
      theta_rad = np.deg2rad(thetas[i])
      # Calculate the steering vector for the current source
      stv = np.pi/2 * array_pos * np.sin(theta_rad)
      # Append the steering vector to the list
      steering_vectors.append(stv.reshape(-1, 1))
    print(thetas)
 
    R=np.zeros((M,K),complex)
    omeg= 2*np.pi*1e9;


    for n in range(K):#K
      #tmp=np.zeros((M,1))
      #print(tmp)
      tmp=0
      for a in range(L):        
        tmp = tmp+ np.exp(1j*(omeg*t[n]-steering_vectors[a])) 
        
      noise = (np.random.randn(M,1)) * pruido
      # R[:,n] = tmp + noise
      R[:,n] = (tmp + noise).flatten()
      # print(R)
    Z = np.concatenate([R.T.real.flatten(), R.T.imag.flatten()])

    X_data[m, :] = Z
  
    # y_labels[m] = mapa[thetas[0]]  # <----------------- PARA ETIQUETAS DE INDICES
    for theta in thetas:
     y_labels[m, mapa[theta]] = 1  # etiqueta BINARIA de longitud G
    # x1 = [0.01, 1, 10, 100]
    # y1 = [49.23, 91.23, 99.99, 99.99]
    
    # plt.plot(np.real(R[0,:]))
    
    # plt.xlim(0,100)
  # plt.plot(x1, y1, marker='o', linestyle='-', color='blue', label=r'10 épocas')  
  # plt.show
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
      def __init__(self, input_size=2*M*K, output_size=clases):
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

M=16
K=100
num_samples=100
SNR_dB=0.1;
precision=1
lr=0.0001
fs = 10e9;
L=2
# t = (0:1/fs:1e-6);
t= np.arange(0,1e-6,1/fs);

angles_de2 = np.arange(-60, 60, precision)
angles_de = np.arange(-60, 60, 1) ### <------- para datos de prueba
clases= len(angles_de2)
print(angles_de)
model,device=modelo(M,K,clases) ###### <-------- MODELO
# 2. Carga los pesos
model.load_state_dict(torch.load("modelo_vs_cnn.pth"))

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
print(result)

ruidos=[-20,-15,-10,-5,0,5,10,15,20,25,30]
ruidos1 =[5,10]
Pglobal=[]
absGlobal=[]
for ss in ruidos:
    X_datax, y_labels = datos1_2(result,array_po,SNR_dB=ss,K=K,num_samples=num_samples,M=M,mapa=mapas,L=L,t=t)
    train_loader, val_loader, X_train, X_val, y_train, y_val=dataset_(X_datax, y_labels)
    inp=torch.tensor(X_datax, dtype=torch.float32).to(device)

    
    with torch.no_grad():
      output = model(inp)
      # prediction = torch.argmax(output, dim=1)
      prediction = torch.argmax(output, dim=1).cpu().numpy()
      predicted = (output > 0.5).int().cpu().numpy()  # Convierte a 0 o 1

    # print("prediccion:",predicted[0])
    er = []
    erabs=[]
    mapa_inv = {v: k for k, v in mapas.items()}  # inverso: índice → theta
    mapa_inv2 = {v: k for k, v in mapas2.items()}  # inverso: índice → theta
    for k in range(len(predicted)):
        # Encuentra los índices donde hay 1
        indices_p = np.where(predicted[k] == 1)[0]
        indices_y = np.where(y_labels[k] == 1)[0]
        # Convierte a los valores reales de theta
        thetas_reales = [mapa_inv[i] for i in indices_y]
        thetas_p = [mapa_inv2[i] for i in indices_p]
        print(f"predicha: {thetas_p}")
        print(f"real: {thetas_reales}")
        xx= emparejar(thetas_p, thetas_reales)
        print("Parejas:",xx)   
        er2=[]
        erabs2=[]
        for s in range(len(xx)):
          if xx[s][0]!=0:
            fallo= float(abs((xx[s][0]-xx[s][1])/(xx[s][0]+1e-3))*100)
            raiz= float((xx[s][1]-xx[s][0])**2)
            falloabs=float(abs(xx[s][1]-xx[s][0]))
            print("raiz:",raiz)
            er.append(raiz)
            erabs2.append(falloabs)
        if len(er2) > 0:
            prome = np.sqrt(sum(er2) / len(er2))
        else:
            prome = 0  # or some other default value

        if len(erabs2) > 0:
            promeabs = sum(erabs2) / len(erabs2)
        else:
            promeabs = 0  # or some other default value

        #e_relativo= np.mean([fallo,fallo2])
        #print(prome)
        # if prome<1000:
        #     er.append(prome)
            
        erabs.append(promeabs)

    # media = sum(er) / len(er)
    # print('RMSE',er)
    # Pglobal.append(media)

    mediaabs= np.sqrt(sum(er) / len(er))
    absGlobal.append(mediaabs)
    print("RMSE:",mediaabs)

# mmedia= sum(er) / len(er)
# print("Promedio de RMSE:",mmedia)
# Nabs = sum(absGlobal)/len(absGlobal)
# print("Promedio de error absoluto:",Nabs)
print(len(absGlobal))
ruidos2=[-20,-15,-10,-5,0,5,10,15,20,25,30]
plt.plot(ruidos,absGlobal)
plt.yscale("log")
plt.ylabel("RMSE")
plt.xlabel("SNR, [dB]")
plt.show();
