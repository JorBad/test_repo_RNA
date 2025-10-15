## hola 2
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytz
import torch
#PARA EL MODELO
import torch.nn as nn

#entrenamiento
import torch.optim as optim
import matplotlib.pyplot as plt


import os
#import datetime
#from google.colab import drive
import shutil

## Parámetros
#M = 10               # Número de sensores
L = 1                # Número de fuentes
#K = 16               # Snapshots
#SNR_dB = 0.2         # Relación señal-ruido
#num_samples = 150000   # Número total de muestras
lambda_ = 1.0
d = lambda_ / 2      # Separación entre sensores
#precision=5;
#angles_deg = np.arange(0, 180, precision)  # 36 clases

#G = len(angles_deg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 2. Definir ruta base en Drive donde se almacenarán las carpetas
#base_drive_path = r"C:\Users\jorge\Documents\RNA_Ksenales"
base_drive_path = r"\content"

# 3. Obtener la fecha actual en formato ddmmaa
today_str = datetime.now().strftime("%d%m%y")
daily_folder_path = os.path.join(base_drive_path, today_str)

# 4. Crear carpeta del día si no existe
os.makedirs(daily_folder_path, exist_ok=True)


def fecha_hora():
  # Definir la zona horaria de Ciudad de México
  zona_mex = pytz.timezone('America/Mexico_City')

  now = datetime.now(zona_mex)
  #print(now)
  now_str = now.strftime("%Y_%m_%d_%H%M%S")
  nametext = f"epocas_DeepMLP_{now_str}.txt"
  return nametext

def datos(angles_deg,array_pos,SNR_dB,K,num_samples,M,mapa):
  # Inicialización
  X_data = np.zeros((num_samples, 2*M*K))  # Entradas reales
  y_labels = np.zeros((num_samples,), dtype=int)  # Etiquetas de clase (índices)
  for n in range(num_samples):
      # Ángulo verdadero (una clase aleatoria)
      #theta_deg = np.random.choice(angles_deg)
      theta_deg= angles_deg[n]
      #print(theta_deg)
      theta_rad = np.deg2rad(theta_deg)
      #print(f"Etapa {n}")
      # Señal incidente (misma para todos los snapshots)
      alpha = 1#np.random.randn()  # amplitud aleatoria real
      steering_vector = np.exp(1j  * np.pi/2 * array_pos * np.sin(theta_rad))

      # Señales incidentes con fase aleatoria para cada snapshot
      signal_snapshots = []
      for _ in range(K):
          random_phase = np.exp(1j * 2 * np.pi * np.random.rand())
          s = alpha * random_phase * steering_vector
          # Añadir ruido complejo
          noise = (np.random.randn(M) + 1j * np.random.randn(M)) * SNR_dB
          snapshot = s + noise
          #print(f"Añado ruido {K}")
          signal_snapshots.append(snapshot)

      # Matriz X: K snapshots x M sensores
      X = np.vstack(signal_snapshots)  # K x M
      # Entrada real: Re(X^T), Im(X^T) → (K x M) x 2 → vector de tamaño 2MK
      Z = np.concatenate([X.T.real.flatten(), X.T.imag.flatten()])

      X_data[n, :] = Z
      #y_labels[n] = np.where(angles_deg == theta_deg)[0][0]
      y_labels[n] = mapa[theta_deg]

      # # Tomar el primer snapshot del primer sample
      # snapshot_0 = signal_snapshots[0]

      # plt.figure()
      # plt.plot(np.real(snapshot_0), label="Parte real")
      # plt.plot(np.imag(snapshot_0), label="Parte imaginaria")
      # plt.title("Snapshot 0 - Señal compleja")
      # plt.xlabel("Índice del sensor")
      # plt.ylabel("Amplitud")
      # plt.legend()
      # plt.grid(True)
      # plt.show()
  return X_data, y_labels

def datos1_2(angles_deg,array_pos,SNR_dB,K,num_samples,M,mapa,L):
  # Inicialización
  X_data = np.zeros((num_samples, 2*M*K))  # Entradas reales

  y_labels = np.zeros((num_samples,len(mapa)), dtype=int)  # Etiquetas de clase binarias

  #y_labels = np.zeros((num_samples,), dtype=int)  # Etiquetas de clase (índices)

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
      stv = np.exp(1j * np.pi * array_pos * np.sin(theta_rad))
      # Append the steering vector to the list
      # steering_vectors.append(stv)
      steering_vectors.append(stv.reshape(-1, 1))
    #print(thetas)
    # Convert the list of steering vectors to a numpy array
    #steering_vectors_array = np.array(steering_vectors)
    # print(steering_vectors[0])
    # print(steering_vectors[1])

    #alpha = np.random.rand(L,1)  # amplitud aleatoria original --->>  .randn
    alpha= np.ones(L)
    R=np.zeros((M,K),complex)



    for n in range(K):#K
      #tmp=np.zeros((M,1))
      #print(tmp)
      tmp=0
      for a in range(L):
        random_phase = np.exp(1j *2 * np.pi * np.random.rand())
        tmp = tmp+alpha[a] * random_phase * steering_vectors[a]  # <----PARA AMPLITUD ALEATORIA
        #tmp = tmp * random_phase * steering_vectors[a]

      noise = (np.random.randn(M,1) + 1j * np.random.randn(M,1)) * SNR_dB
      R[:,n] = (tmp + noise).flatten()

    Z = np.concatenate([R.T.real.flatten(), R.T.imag.flatten()])

    X_data[m, :] = Z

    #y_labels[m] = mapa[thetas[0]]  # <----------------- PARA ETIQUETAS DE INDICES
    for theta in thetas:
      y_labels[m, mapa[theta]] = 1  # etiqueta BINARIA de longitud G

  return X_data, y_labels

def datos2(angles_deg,array_pos,SNR_dB,K,num_samples,M,mapa,L,t,mapaphi,anglesphi):
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
  for m in range(num_samples):
    stv=np.zeros((M,1))
    #print(stv)
    steering_vectors.clear() # Clear for each sample
    for i in range(L):
      # Ángulo verdadero (una clase aleatoria)
      thetas[i] = np.random.choice(angles_deg)
      theta_rad = np.deg2rad(thetas[i])
      phis[i] = np.random.choice(anglesphi)
      phi_rad = np.deg2rad(phis[i])
      # steering vector para lambda/4
      # stv = np.pi/2 * array_pos * np.sin(theta_rad)
      # steering vector para lambda/2
      stv = np.pi * array_pos * np.sin(theta_rad)*np.cos(phi_rad)
    #   print(stv.shape)
      # Append the steering vector to the list
      steering_vectors.append(stv.reshape(-1, 1))
    # print(thetas)
 
    R=np.zeros((M*M,K),complex)
    omeg= 2*np.pi*1e9;


    for n in range(K):#K
      tmp=np.zeros((M*M,1))
      #print(tmp)
    #   tmp=0
      for a in range(L):   
        for ant in range(M-1):    
            indice= ant+((M-1)*(ant))
            print(indice)
            tmp[indice:indice+(M)] = tmp[indice: indice+(M)]+ np.exp(1j*(omeg*t[n]-steering_vectors[a])) 
            print("Temporal",tmp)
      noise = (np.random.randn(M*M,1)) * pruido
      # R[:,n] = tmp + noise
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

def modelo2(M,K,clases):
  #clases= 180/precision
  class DeepMLP(nn.Module):
      def __init__(self, input_size=2*M*K, output_size=clases):
          super(DeepMLP, self).__init__()
          self.model = nn.Sequential(
              nn.Linear(input_size, 512),
              # nn.ReLU(),
              # nn.Tanh(),
              nn.Sigmoid(),
              nn.Dropout(0.3),
              nn.Linear(512, 256),
              # nn.ReLU(),
              nn.Sigmoid(),
              # nn.Tanh(),
              nn.Dropout(0.3),
              nn.Linear(256, 128),
              # nn.ReLU(),
              nn.Sigmoid(),
              # nn.Tanh(),
              nn.Linear(128, int(clases))
          )

      def forward(self, x):
          return self.model(x)

  # Instancia del modelo
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #model = MLPClassifier().to(device)
  model = DeepMLP().to(device)
  return model,device

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

def probarModelo(M,K,SNR_dB,precision,num_samples,L,modelo,t):
    model=modelo
    angles_de = np.arange(-60, 60, precision)
    clases= len(angles_de)

    # model,device=modelo(M,K,clases) ###### <-------- MODELO
    # # 2. Carga los pesos
    # model.load_state_dict(torch.load("modelo_2signals.pth"))

    array_po = np.linspace(0, (M-1), M)
    #angles_de = np.arange(0, 90, 5)
    mapas = {valor: idx for idx, valor in enumerate(angles_de)}
    n_total=num_samples
    numdatos= len(angles_de)
    base_count = n_total // numdatos  #
    restantes = n_total % numdatos    #

    # Inicializar la lista de salida
    result2 = []

    # Añadir base_count veces cada elemento
    for angle in angles_de:
        result2.extend([angle] * base_count)

    for i in range(restantes):
        result2.append(angles_de[i])

    # Convertir a array si se desea
    result2 = np.array(result2)
    # print(result)
    np.random.shuffle(result2)


    #X_data, y_labels = datos(result,array_po,SNR_dB=SNR_dB,K=K,num_samples=num_samples,M=M,mapa=mapas)
    Pglobal=[]
    absGlobal=[]
    for ss in range(10):
        # X_datax, y_labels = datos1_2(result2,array_po,SNR_dB=SNR_dB,K=K,num_samples=num_samples,M=M,mapa=mapas,L=L)
        X_datax, y_labels = datos2(result2,array_po,SNR_dB=SNR_dB,K=K,num_samples=num_samples,M=M,mapa=mapas,L=L,t=t)
        train_loader, val_loader, X_train, X_val, y_train, y_val=dataset_(X_datax, y_labels)
        inp=torch.tensor(X_datax, dtype=torch.float32).to(device)

    
        with torch.no_grad():
            output = model(inp)
            # prediction = torch.argmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output > 0.5).int().cpu().numpy()  # Convierte a 0 o 1

        # print("Entradas nuevas:", y_labels)
        # print("Predicción:", predicted)
        #print("Outputs:", output)

        #per_class_acc = ((predicted == y_labels).float().sum(dim=0)) / y_labels.size(0)
        # per_class_acc = ((predicted == y_labels).astype(np.float32).sum(axis=0)) / y_labels.shape[0]

        # macro_acc = per_class_acc.mean()
        # print(macro_acc)
        # print(f"per_class {per_class_acc}")
        # plt.plot(per_class_acc*100)

        # thetas_reales = np.zeros(3)
        # thetas_p = np.zeros(3)
        er = []
        erabs=[]
        mapa_inv = {v: k for k, v in mapas.items()}  # inverso: índice → theta
        for k in range(len(predicted)):
            # Encuentra los índices donde hay 1
            indices_p = np.where(predicted[k] == 1)[0]
            indices_y = np.where(y_labels[k] == 1)[0]
            # Convierte a los valores reales de theta
            thetas_reales = [mapa_inv[i] for i in indices_y]
            thetas_p = [mapa_inv[i] for i in indices_p]
            # print(f"predicha: {thetas_p}")
            # print(f"real: {thetas_reales}")
            xx= emparejar(thetas_p, thetas_reales)
           
            er2=[]
            erabs2=[]
            for s in range(len(xx)):
                if xx[s][0]!=0:
                    fallo= float(abs((xx[s][0]-xx[s][1])/(xx[s][0]+1e-3))*100)
                    falloabs=float(abs(xx[s][1]-xx[s][0]))

                    er2.append(fallo)
                    erabs2.append(falloabs)
            if len(er2) > 0:
                prome = sum(er2) / len(er2)
            else:
                prome = 0  # or some other default value

            if len(erabs2) > 0:
                promeabs = sum(erabs2) / len(erabs2)
            else:
                promeabs = 0  # or some other default value

            #e_relativo= np.mean([fallo,fallo2])
            #print(prome)
            if prome<1000:
                er.append(prome)
            
            erabs.append(promeabs)

        media = sum(er) / len(er)
        # print('Error relativo',media)
        Pglobal.append(media)

        mediaabs= sum(erabs) / len(erabs)
        absGlobal.append(mediaabs)

    mmedia= sum(Pglobal) / len(Pglobal)
    print("Promedio de error relativo:",mmedia)
    Nabs = sum(absGlobal)/len(absGlobal)
    print("Promedio de error absoluto:",Nabs)
    return mmedia,Nabs

def entrenamiento(M,K,SNR_dB,precision,num_samples,lr,L,t):
  cont=0;
  preci = []

  zona_mex = pytz.timezone('America/Mexico_City')
  now2 = datetime.now(zona_mex)
  #print(now)
  now2_str = now2.strftime("%H_%M_%S")

  angles_de = np.arange(0, 90, precision)
  angles_phi = np.arange(0, 360, precision)

  mapas = {valor: idx for idx, valor in enumerate(angles_de)}

  n_total=num_samples
  numdatos= len(angles_de)
  base_count = n_total // numdatos  #
  restantes = n_total % numdatos    #

  # Inicializar la lista de salida
  result = []

  # Añadir base_count veces cada elemento
  for angle in angles_de:
     result.extend([angle] * base_count)

  # Añadir los elementos restantes (para completar los 100)
  # Puedes seleccionar los primeros "restantes" elementos de x
  for i in range(restantes):
    result.append(angles_de[i])

  # Convertir a array si se desea
  result = np.array(result)
  np.random.shuffle(result)

  
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
#   print(result_p)

#   array_po = np.linspace(0, (M-1)/4, M)
  array_po = np.linspace(0, (M-1), M) #para indices

  clases= len(angles_de)
  nametext=fecha_hora() 
  now4_str = now2.strftime("%H_%M_%S")
  print("Comienza la prueba:",now4_str)
  # X_data, y_labels = datos1_2(result,array_po,SNR_dB,K,num_samples,M,mapas,L)

  nX_data = np.empty((0, 3200))
  ny_labels = np.empty((0, 120))  # array vacío con 0 filas y 120 columnas
  ruidos=[5,20,25,30]
  for trainer in ruidos:
    X_data, y_labels = datos2(result,array_po,trainer,K,num_samples,M,mapas,L,t,mapas_phi,result_p)
    # print("Len Xdata:",len(X_data))
    # print("Len ylabels:",len(y_labels))
    nX_data = np.vstack((nX_data, X_data))  
    ny_labels = np.vstack((ny_labels, y_labels))  # apila por filas
    # print(nX_data.shape)
    # SNR_dB=SNR_dB+0.05;
  train_loader, val_loader, X_train, X_val, y_train, y_val=dataset_(nX_data, ny_labels)

  model,device=modelo(M,K,clases) ################################## <-------- MODELO

  valores, cuentas = np.unique(y_labels, return_counts=True)

  y_labels = torch.tensor(y_labels, dtype=torch.float32)
  # for v, c in zip(valores, cuentas):
  #   print(f"Valor {v}: {c} times")

  print(f"-----------comienza el entrenamiento {now2_str}-------")

  # Use a list to store training losses
  train_losses_list = []
  max_val_acc=[0, 0];

  # Hiperparámetros
  ## FUNCIONES DE PERDIDA
  #criterion = nn.CrossEntropyLoss()
  #criterion = torch.nn.MSELoss()
  #criterion = torch.nn.L1Loss()

  criterion = nn.BCEWithLogitsLoss()

  #criterion2 = ErrorAngular(angles_de) # <---Prueba

  # learning_rate = 0.0001
  learning_rate = lr

  ## OPTIMIZADORES
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
  #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


  num_epochs = 100 ####################### <-------------- EPOCAS #####################

  print(f"Prueba con {M} antenas, {num_samples} muestras , Snapshots={K}, Modelo= DeepMLP,  Epocas={num_epochs}, Precision={precision}, SNR={SNR_dB}, Señales= {L}\n")

  # Crear archivo y escribir encabezado
#   with open(nametext, "w") as f:
#       f.write("# DeepMLP Training Log\n")
#       f.write(f"# Muestras={num_samples}, Antenas={M}, Snapshots={K}, Modelo= DeepMLP,  Epocas={num_epochs}, Precision={precision}, Potencia del ruido={SNR_dB}, Señales= {L}\n")
#       f.write("Epoca\tPrecision de entren:(%)\t\tPrecision de val:(%)\t\tError relativo(%)\n")

  #accs = []
  for epoch in range(num_epochs):
      model.train()
      train_loss, correct, total = 0.0, 0, 0
      er = []
      e_absoluto=[]
      for inputs, targets in train_loader:
          inputs, targets = inputs.to(device), targets.to(device)
          targets = targets.float()
          optimizer.zero_grad()


          #print(targets.dtype)
          ## Backpropagation ##
          outputs = model(inputs)
          # print("outputs.dtype:", outputs.dtype)
          # print("targets.dtype:", targets.dtype)
          # print("outputs.shape:", outputs.shape)
          # print("targets.shape:", targets.shape)
          loss = criterion(outputs, targets)
          #print(loss)
          # loss2 = criterion2(outputs, targets) # <---Prueba
          # if cont>20000 and cont<40000:
          #   loss=loss + loss2 # <---Prueba

          perdida=loss
          #print(f"perdida de funcion: {loss2}")
          #print(x_2)
          loss.backward()
          optimizer.step()
          ###     ###          ###

          train_loss += loss.item() * inputs.size(0)
          # Append the loss for each batch to the list
          train_losses_list.append(perdida.item())

          probs = torch.sigmoid(outputs)
          #_, predicted = torch.max(outputs, 1)
          predicted = (probs > 0.5).int()  # Convierte a 0 o 1



          er2 = []
          erAbs = []
          mapa_inv = {v: k for k, v in mapas.items()}  # inverso: índice → theta
          for k in range(len(predicted)):
            # Encuentra los índices donde hay 1
            indices_p = np.where(predicted[k].cpu() == 1)[0]
            indices_y = np.where(y_labels[k].cpu() == 1)[0]
            # Convierte a los valores reales de theta
            thetas_reales = [mapa_inv[i] for i in indices_y]
            thetas_p = [mapa_inv[i] for i in indices_p]
            # print(f"predicha: {thetas_p}")
            # print(f"real: {thetas_reales}")
            xx= emparejar(thetas_p , thetas_reales)
            # print(f"parejas: {xx}")
            # print(xx[1])
            # fallo= float((abs((xx[1][1])-xx[1][0])/(xx[1][1]+1e-3))*100)
          # print(fallo)
            er3=[]
            erabs3=[]
            for s in range(len(xx)):
              if xx[s][0]!=0:
                #fallo= float((abs((xx[s][0])-xx[s][1])/(xx[s][0]+1e-3))*100)
                fallo= float(abs((xx[s][0]-xx[s][1])/(xx[s][0]+1e-3))*100) ## Real - Estimado / Real *100

                falloAbs=float(abs(xx[s][1]-xx[s][0])) ## Estimado-Real
                erabs3.append(falloAbs)

                er3.append(fallo)
            if len(er3) > 0:
                prome = sum(er3) / len(er3)
            else:
                prome = 100  # or some other default value
            if len(erabs3) > 0:
               promeAbs= sum(erabs3) / len(erabs3)
            else:
               promeAbs=100

            if prome<1000:
              er2.append(prome)
            
            erAbs.append(promeAbs)

          # total += targets.size(0)
          # correct += (predicted == targets).sum().item()
          per_class_acc = ((predicted == targets).float().sum(dim=0)) / targets.size(0)

      acc = per_class_acc.mean()*100
      media = sum(er2) / len(er2)
      mediaAbs= sum(erAbs) / len(erAbs)
      #acc = correct / total * 100
      val_loss, val_correct, val_total = 0.0, 0, 0
      model.eval()

      with torch.no_grad():
          for inputs, targets in val_loader:
              inputs, targets = inputs.to(device), targets.to(device)

              targets = targets.float()

              outputs = model(inputs)
              loss = criterion(outputs, targets)
              val_loss += loss.item() * inputs.size(0)

              probs = torch.sigmoid(outputs)

              #_, predicted = torch.max(outputs, 1)
              predicted_val = (probs > 0.5).int()  # Convierte a 0 o 1

              # val_total += targets.size(0)
              # val_correct += (predicted == targets).sum().item()
              per_class_valacc = ((predicted_val == targets).float().sum(dim=0)) / targets.size(0)

      val_acc = per_class_valacc.mean()*100
      # print(f"tipo de dato: {val_acc.type}")
      nuevoVal=val_acc.cpu().float()
      #val_acc = val_correct / val_total * 100
      print(f"Epoca {epoch+1:02d} | Precision de entren: {acc:.2f} | Error relativo: {media:.2f} | Error absoluto: {mediaAbs:.2f}")
      preci.append(val_acc)
      if nuevoVal>max_val_acc[0]:
          max_val_acc=[nuevoVal, epoch+1];

      # Guardar en archivo
    #   with open(nametext, "a") as f:
    #         f.write(f"{epoch+1}\t\t\t{acc:.2f}%\t\t\t\t\t\t{val_acc:.2f}%\t\t\t\t {media:.2f}%\n")

  # Convert the list of training losses to a numpy array after the loop
  ratioerror = np.array(train_losses_list)
  print(f"-----------------------------------------------------")
  now2 = datetime.now(zona_mex)
  now3_str = now2.strftime("%H_%M_%S")
  print("Se acabó la prueba a las:",now3_str)
  x = np.arange(len(preci))
  #plt.plot(x, preci, marker='o', linestyle='-')
  plt.ylabel("Precisión")
  plt.xlabel("Epocas")
  #plt.show()
#   with open(nametext, "a") as f:
#     f.__del__
  #         0             1         2      3      4       5       6       7
  return max_val_acc,ratioerror,nametext,model,X_train, X_val, y_train, y_val,device


M=5;
Kvalues=[1]
K=100
SNR_dB=10
precision=1;
Pvalues=[5]
num_samples=50000
pruebas=1000
lr=0.0001
fs = 10e9;
t= np.arange(0,1e-6,1/fs);
noises=[0.01,0.1,1,2]
signals=[5]
signal=2
scores = []
errores = []
erroresAbs=[]
for Kv in signals:
    red = entrenamiento(M=M, K=K, SNR_dB=SNR_dB, precision=precision, num_samples=num_samples,lr=lr,L=Kv,t=t)
    scores.append(red[0][0])
    # plt.plot(scores)
    prueba=probarModelo(M,K,SNR_dB,precision,pruebas,Kv,red[3],t)
    errores.append(prueba[0])
    erroresAbs.append(prueba[1])

# plt.plot(signals,errores)
# # Añadir título y etiquetas
# plt.title("Errores relativos")
# plt.xlabel("Señales incidentes")
# plt.ylabel("porcentaje de error")
# # plt.xscale("log")
# # Mostrar gráfica
# plt.show()

# plt.plot(signals,erroresAbs)
# plt.title("Errores absolutos")
# plt.xlabel("Señales incidentes")
# plt.ylabel("error [°]")
# # plt.xscale("log")
# # Mostrar gráfica
# plt.show()

name = f"modelo1_2Dp1.pth"

model=red[3]
torch.save(model.state_dict(), name)
