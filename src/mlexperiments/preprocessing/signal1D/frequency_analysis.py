from scipy import signal


def get_scipy_spectogram(x, sample_rate):
    frequencies, times, spectrogram = signal.spectrogram(x, sample_rate)
    return frequencies, times, spectrogram