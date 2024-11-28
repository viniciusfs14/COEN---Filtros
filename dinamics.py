import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import iirfilter, sosfilt
from scipy.fft import fft, fftfreq
import soundfile as sf

filename = 'audio_ruido.mp3'
x, fs = librosa.load(filename, sr=None)  # sr=None para manter a taxa de amostragem original

cutoff_low = 4500  # Frequência de corte do filtro passa-baixa em Hz
cutoff_high = 5500  # Frequência de corte do filtro passa-alta em Hz
order = 100  # Ordem dos filtros

# Projete o filtro passa-baixa
sos_low = iirfilter(N=order, Wn=cutoff_low, btype='low', fs=fs, ftype='butter', output='sos')

# Projete o filtro passa-alta
sos_high = iirfilter(N=order, Wn=cutoff_high, btype='high', fs=fs, ftype='butter', output='sos')

# Aplicar os filtros ao sinal
y_low = sosfilt(sos_low, x)
y_high = sosfilt(sos_high, x)

# Combinar os resultados
y = y_high - y_low

# Salvar o sinal filtrado em um novo arquivo
output_filename = 'musica_filtrada.mp3'
sf.write(output_filename, y, fs)

# Função para calcular e plotar a FFT
def plot_fft(signal, fs, title):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)
    plt.plot(xf[:N // 2], np.abs(yf[:N // 2]))
    plt.title(title)
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()

# Plotar o sinal original e filtrado
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
librosa.display.waveshow(x, sr=fs)
plt.title('Sinal Original')

plt.subplot(2, 2, 2)
plot_fft(x, fs, 'FFT do Sinal Original')

plt.subplot(2, 2, 3)
librosa.display.waveshow(y, sr=fs)
plt.title('Sinal Filtrado')

plt.subplot(2, 2, 4)
plot_fft(y, fs, 'FFT do Sinal Filtrado')

plt.tight_layout()
plt.show()
