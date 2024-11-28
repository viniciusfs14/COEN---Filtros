import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as signal 

fs = 1000
cutoff = 100
numtaps = 100

b = signal.firwin(numtaps, cutoff/(0.5*fs), window='hamming')
t = np.arange(0, 1.0, 1/fs)
x = 0.5*np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*200*t)
y = signal.lfilter(b, 1.0, x)

#plt.plot(t, x, label="Sinal Original", color='blue')
plt.plot(t, y, label="Sinal Filtrado", color='red')

plt.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Sinal Filtrado', fontsize=25)
plt.xlabel('Tempo [s]', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.show()