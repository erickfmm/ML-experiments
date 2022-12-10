def pre_emphasis(signal, alpha):
    new_signal = []
    new_signal.append(signal[0] - alpha * signal[0])
    for i in range(1, len(signal)):
        new_signal.append(signal[i] - alpha * signal[i-1])
        # new_signal.append(1 - alpha * (1/signal[i]))
    return new_signal


import pywt
from scipy.signal import butter, lfilter


def denoise_wavelet(signal, type_wavelet='db5'):
    cA, cD = pywt.dwt(signal, type_wavelet)
    return cA


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')  # TODO: test
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
