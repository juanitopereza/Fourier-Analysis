
# coding: utf-8

# In[1]:

from scipy.io.wavfile import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[2]:

#Definimos la función que efectua la transformada rápida de Fourier. Devuelve array con coeff complejos.
def FFT(x):
    N = len(x)
    N_min = min(N, 32)
    
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] / 2]
        
        X_odd = X[:, X.shape[1] / 2:]
        
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()


# In[3]:

#Definimos la función que efectua la transformada inversa rápida de Fourier. Devuelve array con coeff complejos.
def IFFT(x):
    N = len(x)
    N_min = min(N, 32)
    
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(2j * np.pi * n * k / N_min)
    X = (1./N)*np.dot(M, x.reshape((N_min, -1)))
    
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] / 2]
        
        X_odd = X[:, X.shape[1] / 2:]
    
        factor = np.exp(1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()


# In[4]:

#Función que filtra la frecuencia con mayor amplitud.
def MataMayor(x):
    y = np.copy(x)
    y[np.where(abs(x) == max(abs(x)))] = 0+0j    
    return y


# In[5]:

# Función que filtra frecuencias f > cutoff.
def PasaBajas(x, freqs, cutoff):
    y = np.copy(x)
    y[np.where(freqs > cutoff)] = 0+0j    
    return y    


# In[6]:

#Lee Datos de los archivos de audio. Extrae datos de "sample rate", número de datos y efectúa "Zero_padding" para poder usar FFT.
DatosDo = read('Do.wav')[1]
f_Do = float(read('Do.wav')[0])
N_Do = len(DatosDo)
Do = np.append(DatosDo,np.zeros(2**16 - N_Do))
dt_Do = 1.0/f_Do

DatosSol = read('Sol.wav')[1]
f_Sol = float(read('Sol.wav')[0])
N_Sol = len(DatosSol)
Sol = np.append(DatosSol,np.zeros(2**16 - N_Sol))
dt_Sol = 1.0/f_Sol


# In[7]:

fou = FFT(Do) #Transformada de la señal Do original
freqs = np.arange(0,len(Do))*f_Do/len(Do) #Reconstruye frecuencias

filtered_1 = MataMayor(fou) #Aplica filtro pico 
filtered_2 = PasaBajas(fou, freqs, 1000) #Aplica filtro pasabajas

t = np.linspace(0,len(Do)/f_Do,len(Do))

#Grafica señal Do original en espacio de frecuencias junto con los filtros aplicados.
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 2])

plt.figure(figsize = (10,12))
ax = plt.subplot(gs[0, :])
plt.plot(freqs, abs(fou))
plt.xlim(0,4e3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('$f\ [Hz]$')
plt.title(u'$Frecuencias\ Señal\ Original\ Do$', fontsize=20)

ax = plt.subplot(gs[1, 0])
plt.plot(freqs, abs(filtered_1))
plt.xlim(0,4e3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('$f\ [Hz]$')
plt.title('$Filtro\ Frecuencia\ Pico$')

ax = plt.subplot(gs[1, 1])
plt.plot(freqs, abs(filtered_2))
plt.xlim(0,4e3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('$f\ [Hz]$')
plt.title('$Filtro\ Pasa\ Bajas$')


plt.savefig('DoFiltros.pdf')
plt.close()


# In[8]:

f_DoSol = 3.*f_Do/2. #Nueva "sample rate" para tranformar Do a Sol
freqs_DoSol = np.arange(0,len(Do))*f_DoSol/len(Do)

fou_sol = FFT(Sol) # TRansformada de Sol
freqs_sol = np.arange(0,len(Sol))*f_Sol/len(Sol) #Recupera frecuencias 

#Grafica Sol artificial (DO -> SOl) vs. Sol original en espacio de frecuencias.
plt.plot(freqs_sol,abs(fou_sol),'r', label = '$Sol$')
plt.plot(freqs_DoSol,abs(fou),'b', label = '$Do \mapsto Sol$')
plt.legend()
plt.xlim(0,4e3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('$f\ [Hz]$')
plt.title('$Do \mapsto Sol\ vs.\ Sol$')

plt.savefig('DoSol.pdf')
plt.close()


# In[9]:

#Recupera Señales usando la transformada inversa y las guarda en formato de audio wav.
write('Do_pico.wav',f_Do,np.real(IFFT(filtered_1))[:N_Do].astype(np.int16))
write('Do_pasabajos.wav',f_Do,np.real(IFFT(filtered_2))[:N_Do].astype(np.int16))
write('DoSol.wav',f_DoSol,np.real(IFFT(fou))[:N_Sol].astype(np.int16))

