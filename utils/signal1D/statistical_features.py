import numpy as np

def statistical_features(coefficients):
	mean = np.mean(coefficients)
	var = np.var(coefficients)
	median = np.median(coefficients)
	maximmum = np.max(coefficients)
	minimmum = np.min(coefficients)
	return [mean, var, median, maximmum, minimmum]

def statistical_paper_pro(coefficients):
	mean = np.mean(coefficients)
	var = np.var(coefficients)
	maximmum = np.max(coefficients)
	return [mean, var, maximmum]

def statistical_paper_pro_4(coefficients):
	mean = np.mean(coefficients)
	var = np.var(coefficients)
	maximmum = np.max(coefficients)
	minimmum = np.min(coefficients)
	return [mean, var, maximmum, minimmum]

def statistical_slope(coefficients, ms_step):
	rising_values = []
	rising_dur = []
	falling_values = []
	falling_dur = []
	rising_its = 0
	falling_its = 0
	for i_coef in range(1, len(coefficients)):
		if(coefficients[i_coef] - coefficients[i_coef-1]) > 0: #rising
			rising_values.append(np.abs(coefficients[i_coef] - coefficients[i_coef-1]))
			if falling_its > 0:
				falling_dur.append(ms_step*falling_its)
			falling_its = 0
			rising_its += 1
		else: #falling
			falling_values.append(np.abs(coefficients[i_coef] - coefficients[i_coef-1]))
			if rising_its > 0:
				rising_dur.append(ms_step*rising_its)
			rising_its = 0
			falling_its += 1
	features = []
	features.append(np.mean(rising_dur))
	features.append(np.median(rising_dur))
	features.append(np.max(rising_dur))

	features.append(np.mean(falling_dur))
	features.append(np.median(falling_dur))
	features.append(np.max(falling_dur))


	features.append(np.mean(rising_values))
	features.append(np.median(rising_values))
	features.append(np.max(rising_values))


	features.append(np.mean(falling_values))
	features.append(np.median(falling_values))
	features.append(np.max(falling_values))

	interquartile_falling_values = np.sort(falling_values)[int(len(falling_values)/4):int(len(falling_values)/4*3)]
	interquartile_rising_values = np.sort(rising_values)[int(len(rising_values)/4):int(len(rising_values)/4*3)]
	interquartile_falling_dur = np.sort(falling_dur)[int(len(falling_dur)/4):int(len(falling_dur)/4*3)]
	interquartile_rising_dur = np.sort(rising_dur)[int(len(rising_dur)/4):int(len(rising_dur)/4*3)]

	features.append(np.max(falling_values)-np.min(falling_values))
	features.append(np.max(rising_values)-np.min(rising_values))

	features.append(np.max(falling_dur)-np.min(falling_dur))
	features.append(np.max(rising_dur)-np.min(rising_dur))

	return features







#derivatives in first, second and third order
def get_feature_vector31_audio(signal):
	features = []
	features.append(np.median(signal))
	features.append(np.mean(signal))
	features.append(np.std(signal))
	features.append(np.min(signal))
	features.append(np.max(signal))
	features.append(np.max(signal)-np.min(signal))

	ratios = []
	first_differences = []
	last_sample = 0
	first = True
	for sample in signal:
		if first:
			last_sample = sample
			first = False
			continue
		first_differences.append(np.absolute(sample - last_sample))
		ratios.append(np.absolute(sample / last_sample))
		last_sample = sample

	features.append(np.min(first_differences))
	features.append(np.max(first_differences))

	features.append(np.min(ratios))
	features.append(np.max(ratios))

	#########################################################
	#second step
	features.append(np.median(first_differences))
	features.append(np.mean(first_differences))
	features.append(np.std(first_differences))
	features.append(np.max(first_differences)-np.min(first_differences))
	#features.append(np.min(first_differences))
	#features.append(np.max(first_differences))

	second_step_ratios = []
	second_differences = []
	last_sample = 0
	first = True
	for sample in first_differences:
		if first:
			last_sample = sample
			first = False
			continue
		second_step_ratios.append(np.absolute(sample / last_sample))
		second_differences.append(np.absolute(sample - last_sample))
		last_sample = sample

	features.append(np.min(second_differences))
	features.append(np.max(second_differences))

	features.append(np.min(second_step_ratios))
	features.append(np.max(second_step_ratios))
	#############################################################################3
	#third step
	features.append(np.median(second_differences))
	features.append(np.mean(second_differences))
	features.append(np.std(second_differences))
	features.append(np.max(second_differences)-np.min(second_differences))
	#features.append(np.min(second_differences))
	#features.append(np.max(second_differences))

	third_step_ratios = []
	third_differences = []
	last_sample = 0
	first = True
	for sample in second_differences:
		if first:
			last_sample = sample
			first = False
			continue
		third_differences.append(np.absolute(sample - last_sample))
		third_step_ratios.append(np.absolute(sample / last_sample))
		last_sample = sample

	features.append(np.min(third_differences))
	features.append(np.max(third_differences))

	features.append(np.min(third_step_ratios))
	features.append(np.max(third_step_ratios))
	features.append(np.max(third_differences)-np.min(third_differences))
	"""
	Median, mean, standard deviation, minimum, maximum, minimum, maximum ratio, and maximum and minimum difference
	first order difference and two order difference and 6
	Median, mean, standard deviation, minimum, maximum, minimum, maximum ratio, and maximum and minimum difference
	"""
	#################
	fft = np.fft.fft(signal)
	features.append(np.absolute(np.median(fft)))
	features.append(np.absolute(np.mean(fft)))
	features.append(np.absolute(np.std(fft)))
	features.append(np.absolute(np.max(fft)))
	features.append(np.absolute(np.min(fft)))
	#features.append(np.absolute(np.min(fft)+np.max(fft))) #ni idea, dice "maximum and minimum" y no sé que es y es una sola cosa o no da la cantidad ._.
	features.append(np.absolute(np.max(fft)-np.min(fft)))
	"""
	Frequency
	Median, mean, standard deviation, maximum, minimum, maximum and minimum (¿6?)
	"""
	return features
