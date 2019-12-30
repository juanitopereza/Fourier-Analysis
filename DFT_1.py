import numpy as np
import matplotlib.pyplot as plt
#%%
Nsamp = 100
T = 2*np.pi
x = np.linspace(0,T, Nsamp+1)
sig_1 = 3*np.cos(x_2) + 2*np.cos(3*x_2) + np.cos(5*x_2)
sig_2 = np.sin(x_2) + 2*np.sin(3*x_2) + 3*np.sin(5*x_2)
sig_3 = 5*np.sin(x_2) + 2*np.cos(3*x_2) + np.sin(5*x_2)

#%%

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(311)
ax.scatter(x, sig_1)
ax1 = fig.add_subplot(312)
ax1.scatter(x, sig_2)
ax2 = fig.add_subplot(313)
ax2.scatter(x, sig_3)

fig.suptitle("Signals")

#%%

def fourier(signal):
    N = len(signal) - 1
    DFTz = np.zeros(N, complex)
    for n in range(N):
        zsum = 0 + 0j
        for k in range(N):
            zsum += signal[k]*np.exp(-2j*np.pi*n*k/N)
        DFTz[n] = zsum*(1.0/np.sqrt(2*np.pi))
    return DFTz

#%%

dft_1 = fourier(sig_1)
dft_2 = fourier(sig_2)
dft_3 = fourier(sig_3)

#%%

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(311)
ax.plot(np.abs(dft_1)**2)
ax.set_xlim(0,10)
ax1 = fig.add_subplot(312)
ax1.plot(np.abs(dft_2)**2)
ax1.set_xlim(0,10)
ax2 = fig.add_subplot(313)
ax2.plot(np.abs(dft_3)**2)
ax2.set_xlim(0,10)

fig.suptitle("Power spectrum")

#%%

#Proporciones entre ls componetes de fourier

np.abs(dft_1[1])/np.abs(dft_1[3])
np.abs(dft_1[1])/np.abs(dft_1[5])

np.abs(dft_2[1])/np.abs(dft_2[3])
np.abs(dft_2[1])/np.abs(dft_2[5])

np.abs(dft_3[1])/np.abs(dft_3[3])
np.abs(dft_3[1])/np.abs(dft_3[5])

#%%
recon_1 = 0.0
for k in range(Nsamp):
    recon_1 += dft_1[k].real*np.cos(k*x)

dft_1[1]

#%%
plt.plot(x,recon_1)
plt.plot(x,sig_1)
