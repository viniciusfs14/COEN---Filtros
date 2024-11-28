import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import iirfilter, sosfilt
from scipy.fft import fft, fftfreq
import soundfile as sf

filename = 'audio_ruido.mp3'
x, fs = librosa.load(filename, sr=44100)  # sr=None para manter a taxa de amostragem original

N = len(x)
yf = fft(x)
xf = fftfreq(N, 1 / fs)
plt.plot(xf[:N // 2], np.abs(yf[:N // 2]))

plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()