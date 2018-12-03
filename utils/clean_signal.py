from deps.Voice_Activity_Detector import vad



def getVoiceSignal(signal, sr, nFFT=512, win_length=0.025, hop_length=0.01, theshold=0.99, thresholdProbVoice = 0.5):
    probSignal=vad.VAD(signal, sr, nFFT=nFFT, win_length=win_length, hop_length=hop_length, theshold=theshold)
    
    t = int(hop_length*sr)
    signalVoice = []
    for i in range(len(probSignal)):
        if i > 0 and i < len(probSignal)-1 and probSignal[i] > thresholdProbVoice:
            signalVoice.extend(signal[t*i:t*(i+1)])
    return (probSignal, signalVoice)