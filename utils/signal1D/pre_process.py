import numpy as np

def normalize_absmax(signal):
	maxim = np.max(signal)
	minim = np.min(signal)
	absolute_max = np.abs(maxim) if np.abs(maxim) > np.abs(minim) else np.abs(minim)
	for i in range(0, len(signal)):
		signal[i] = signal[i] / float(absolute_max)
	return signal