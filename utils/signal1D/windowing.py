from scipy.signal import get_window

def framing(signal, size, step):
	frames = []
	start = 0
	end = size
	while end < len(signal):
		frames.append(signal[start:end])
		start += step
		end += step
	return frames

def windowing_time(signal, window_type):
	w = get_window(window_type, len(signal))
	new_signal = [signal[i]*w[i] for i in range(0, len(w))]
	return new_signal

def framing_with_window(signal, size, step, window_type=None):
    frames = []
    start = 0
    end = size
    w = [1 for i in range(size)]
    if window_type is not None:
        w = get_window(window_type, len(signal))
    while end < len(signal):
        if window_type is None:
            frames.append(signal[start:end])
        else:
            sub_signal = signal[start:end]
            new_signal = [sub_signal[i]*w[i] for i in range(0, len(w))]
            frames.append()
        start += step
        end += step
    return frames