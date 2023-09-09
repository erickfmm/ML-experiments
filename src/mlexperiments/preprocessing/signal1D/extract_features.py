import mlexperiments.deps.frequency_estimator as freqEstimator
from librosa import feature
import numpy as np


def energy(signal):
	energy = 0
	for x in signal:
		energy += np.abs(x)*np.abs(x)
	return energy


def fundamental_frequency(signal, fs):
	return freqEstimator.freq_from_crossings(signal, fs)


def zero_crossing_rate(signal):
	zcr = 0
	for i in range(1, len(signal)):
		zcr += 0.5 * (sign(signal[i]) - sign(signal[i-1]))
	return zcr
	#return feature.zero_crossing_rate(np.array(signal))


def sign(x):
	if x >= 0:
		return 1
	else:
		return -1


def get_mfcc(signal, rate, number):
	return feature.mfcc(np.array(signal), rate,n_mfcc=number)
