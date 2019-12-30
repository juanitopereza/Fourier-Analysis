from scipy.io.wavfile import *
import numpy as np
import matplotlib.pyplot as plt
#%%

sample_rate_do = read('Do.wav')[0]
datos_do = read('Do.wav')[1]
sample_rate_sol = read('Sol.wav')[0]
datos_sol = read('Sol.wav')[1]

#%%

FFT_do = np.fft.fft(datos_do)
freqs = np.fft.fftfreq(len(datos_do), d=1.0/sample_rate_do)

plt.plot(freqs, np.abs(FFT_do))
plt.title("$Do\ original$")
plt.savefig("Do.pdf")
plt.close()

fft_do_menos_mayor = FFT_do.copy()
fft_do_menos_mayor[np.where(np.abs(fft_do_menos_mayor) == np.max(np.abs(fft_do_menos_mayor)))] = 0.+0.j

plt.plot(freqs, np.abs(fft_do_menos_mayor))
plt.title("$Do\ filtro\ mayor\ frecuencia$")
plt.savefig("Do_filtro_mayor_frecuancia.pdf")
plt.close()

fft_do_pasa_bajos = FFT_do.copy()
fft_do_pasa_bajos[np.where(np.abs(freqs) > 1000.)] = 0.+0.j

plt.plot(freqs, np.abs(fft_do_pasa_bajos))
plt.title("$Do\ filtro\ pasa\ bajos$")
plt.savefig("Do_filtro_pasa_bajas.pdf")
plt.close()

#%%

freqs_do_sol = np.fft.fftfreq(len(datos_do), d=1.0/(sample_rate_do*(391./260.)))

len(freqs_do_sol)
FFT_sol = np.fft.fft(datos_sol)
freqs_sol = np.fft.fftfreq(len(datos_sol), d=1.0/sample_rate_sol)
len(freqs_sol)

plt.plot(freqs_sol, np.abs(FFT_sol), label="$Sol\ original$")
plt.plot(freqs_do_sol, np.abs(FFT_do), label="$Sol-Do$")
plt.legend()
plt.title("$Sol\ original\ vs.\ Do-Sol$")
plt.savefig("Do-Sol.pdf")
plt.close()
#%%
