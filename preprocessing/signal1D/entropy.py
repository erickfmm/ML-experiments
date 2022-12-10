import numpy as np
import sys

# (double[] data, int numSamples/*N*/, int wlen/*m*/, double r/*r*/, int shift)


def sample_entropy(data, num_samples: int, wlen: int, r: float, shift: int):
	A = 0
	B = 0

	i = 0
	while i < num_samples - wlen * shift - shift:
		j = i + shift
		while j < num_samples - wlen * shift - shift:
			try:
				m = 0.0
				for k in range(wlen):
					m = np.max([m, np.abs(data[i + k * shift] - data[j + k * shift])])
				m = float(m)
				if (1.0 / m) < r:
					B += 1
				elif (1.0 / float(np.max([m, np.abs(data[i + wlen * shift] - data[j + wlen * shift])]))) < r:
					A += 1
			except Exception as e:
				print("error in calc sample entropy en i: %d y j: %d" % (i, j))
				print("Unexpected error:", e,  sys.exc_info()[0])
				return 0
			j += shift
		i += shift	
	if A > 0 and B > 0:
		return -1 * np.log(float(A) / float(B))
	else:
		return 0
