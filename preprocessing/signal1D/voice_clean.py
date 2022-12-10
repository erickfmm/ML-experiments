from deps.Voice_Activity_Detector import vad


def get_voice_signal(signal,
                     signal_rate,
                     n_fft=512,
                     win_length=0.025,
                     hop_length=0.01,
                     threshold=0.99,
                     threshold_probability_of_voice=0.5):
    probability_signal = vad.VAD(signal,
                                 signal_rate,
                                 nFFT=n_fft,
                                 win_length=win_length,
                                 hop_length=hop_length,
                                 theshold=threshold)
    
    t = int(hop_length * signal_rate)
    signal_with_only_voice = []
    for i in range(len(probability_signal)):
        if 0 < i < len(probability_signal)-1 and probability_signal[i] > threshold_probability_of_voice:
            signal_with_only_voice.extend(signal[t*i:t*(i+1)])
    return probability_signal, signal_with_only_voice
