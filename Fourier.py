import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import *
##Almacenamiento de los datos##
data1 = np.genfromtxt("signalSuma.dat")
data2 = np.genfromtxt("signal.dat")
#print len(data1[:,1])
N = 2048
dt = abs(data1[1,0]-data1[2,0])
##subplots con el grafico para cada senal##
plt.figure()
plt.subplot(2,1,1)
plt.title("signalSuma")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
plt.plot(data1[:,0],data1[:,1])

plt.subplot(2,1,2)
plt.title("signal")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
plt.plot(data2[:,0],data2[:,1])

plt.savefig("signal_subplots.pdf")

##implementacion de la transformada discreta de fourier##
def Fourier(N,data):
	amplitudes = []
	for n in range(N):
		suma = 0
		for k in range(N):
			suma += data[k]*np.exp(-2j*np.pi*k*(n*1.0/N))
		print(suma)
		amplitudes.append(suma)
	return np.asarray(amplitudes)

res = abs(Fourier(N,data1[:,1]))
#res2 = abs(Fourier(N,data2[:,1]))
freq = fftfreq(N,d=dt)

sci_res = fft(data1[:,1])
plt.figure()
plt.plot(freq,res)
plt.xlim(0,3000)
plt.show()
##graficos de la transformada de Fourier de ambas senales##
#print (len(res),len(res2),len(fftfreq(N,d=dt)))

#plt.figure()
#plt.plot(res)
#plt.plot(fftfreq(N,d=dt),res2)
#plt.show()
##espectrograma de las senales##
plt.figure()
plt.subplot(2,1,1)
plt.specgram(data1[:,1])

plt.subplot(2,1,2)
plt.specgram(data2[:,1])
plt.show()
