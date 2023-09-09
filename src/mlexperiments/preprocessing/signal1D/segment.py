# gsr, 6, 256

def get_middle_seconds(signal, seconds, frequency):
	half_seconds = seconds/2
	min_ = int(len(signal)/2-(frequency*half_seconds))
	max_ = int(len(signal)/2+(frequency*half_seconds))
	return signal[min_:max_]


def get_exact_index_segment(signal, min_length, max_length):
	if len(signal) < max_length:
		raise Exception("not enough elements in signal")
	length = max_length - min_length
	new_signal = signal[min_length:max_length]
	if len(new_signal) == length:
		return new_signal
	else:
		raise Exception("different length in signal")
