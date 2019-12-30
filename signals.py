import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import *
#%%

suma_data = np.loadtxt('signalSuma.dat')
signal_data = np.loadtxt('signal.dat')

#%%
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

fig.suptitle("madafaka")

ax1.plot(signal_data[:,0], signal_data[:,1])
ax1.set_xlabel("time")
ax1.set_ylabel("amplitud")
ax2.plot(suma_data[:,0], suma_data[:,1])
ax2.set_xlabel("time")
ax2.set_ylabel("amplitud")

plt.show()

#%%

def DFT_mclaren(x):
    N = len(x)
    n = np.arange(N)
    k = np.transpose([n]) #puedo usar reshape(N,1) aqui
    matrix = np.exp(-2j*np.pi*n*k/N)
    return np.dot(matrix, x)

#%%

def DFT_williams(x):
    N = len(x)
    trans = []
    for k in range(N):
        sum = 0
        for n in range(N):
            sum += x[n]*np.exp(-2j*np.pi*n*k/N)
        trans.append(sum)
    return np.array(trans)

#%%

N = len(signal_data[:,1])
dt = signal_data[1,0]-signal_data[0,0]
fft_signal = fft(signal_data[:,1])[:int(N/2)]
fft_signal_freqs= fftfreq(N, d = dt)[:int(N/2)]
#shifted = fftshift(fft_signal_freqs)
#%%

plt.plot(fft_signal_freqs,abs(fft_signal))
#%%

N = len(signal_data[:,1])
dt = signal_data[1,0]-signal_data[0,0]
DFT_mclaren_signal = DFT_mclaren(signal_data[:,1])[:int(N/2)]
freqs_mac_signal = np.arange(N/2)/(N*dt)
#%%

plt.plot(freqs_mac_signal, abs(DFT_mclaren_signal))

#%%
N = len(signal_data[:,1])
dt = signal_data[1,0]-signal_data[0,0]
DFT_williams_signal = DFT_williams(signal_data[:,1])[:int(N/2)]
freqs_wil_signal = np.arange(N/2)/(N*dt)
#%%

plt.plot(freqs_wil_signal, abs(DFT_williams_signal))
#%%

Fs = 1.0/dt

plt.specgram(signal_data[:,1], Fs=Fs)
