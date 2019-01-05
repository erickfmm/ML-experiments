import csv
import numpy as np
import scipy.io.wavfile as wav
import load_data.data_inside.not_shared.utils_audio_emotion as load_audio_emotions

def get_characterizing_data():
    import pickle
    print("to begin")
    out_data = {'berlin': [], 'savee': [], 'ravdess': [], 'tess': []}
    berlin = load_audio_emotions.get_berlin()
    savee = load_audio_emotions.get_savee()
    ravdess = load_audio_emotions.get_ravdess()
    tess = load_audio_emotions.get_tess()
    datasets = {'berlin': berlin, 'savee': savee, 'ravdess': ravdess, 'tess': tess}
    fieldnames = ['dataset', 'tag', 'type', 'len', 'voice_len', \
     'secs_total', 'secs_voice', 'secs_difference', 'sr', 'freqs', 'times', \
     'voice_freqs', 'voice_times', 'signal_mean', 'signal_std', \
     'voice_mean', 'voice_std', 'spec_mean', 'spec_std', \
     'voice_spec_mean', 'voice_spec_std']
    all_csv_file = open('stats_of_data/all.csv', 'w')
    all_writer = csv.DictWriter(all_csv_file, fieldnames=fieldnames)
    errors_file = open("stats_of_data/errors.txt", "w")
    for dataset in datasets:
        print("to: ", dataset)
        print("len: ", len(datasets[dataset]))
        csv_file = open('stats_of_data/'+dataset+'.csv', 'w')
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for f in datasets[dataset]:
            try:
                (sr, signal) = wav.read(f['ruta'])
                frequencies, times, spectrogram = \
                 utils.get_spectrum.get_scipy_spectogram(np.asarray(signal), sr)
                voice_prob, voice_signal = utils.clean_signal.getVoiceSignal(signal, sr)
                voice_frequencies, voice_times, voice_spectrogram = \
                 utils.get_spectrum.get_scipy_spectogram(np.asarray(voice_signal), sr)
                secs_total = float(len(signal))/float(sr)
                secs_voice = float(len(voice_signal))/float(sr)
                d = {'dataset': dataset, 'tag': f['tag'], 'type': f['type'], 'len': len(signal), \
                 'voice_len': len(voice_signal), 'secs_total': secs_total, \
                 'secs_voice': secs_voice, 'secs_difference': secs_total - secs_voice, 'sr': sr, 'freqs': len(frequencies), \
                'times': len(times), 'voice_freqs': len(voice_frequencies), 'voice_times': len(voice_times), \
                'signal_mean': np.mean(signal), 'signal_std': np.std(signal), \
                'voice_mean': np.mean(voice_signal), 'voice_std': np.std(voice_signal), \
                'spec_mean': np.mean(spectrogram), 'spec_std': np.std(spectrogram), \
                'voice_spec_mean': np.mean(voice_spectrogram), 'voice_spec_std': np.std(voice_spectrogram)}
                writer.writerow(d)
                all_writer.writerow(d)
                out_data[dataset].append(d)
            except:
                errors_file.write(dataset+","+f['tag']+","+f['ruta'])
                errors_file.flush()
        csv_file.flush()
        csv_file.close()
        print("finish ", dataset)
        print("-----------------------------")
        print("-----------------------------")
    errors_file.flush()
    errors_file.close()
    all_csv_file.flush()
    all_csv_file.close()
    print("to pickle")
    with open("stats_of_data/datas.pkl", "wb") as pickle_file:
        pickle.dump(out_data, pickle_file)
    print("ended")
