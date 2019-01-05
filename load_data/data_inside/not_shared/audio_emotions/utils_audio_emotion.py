import os

def get_berlin(data_path="./train_data/not_shared/AudioEmotion/Berlin"):
    data = []
    for root, dirs, files in os.walk(data_path):
        if root.find('male\\') > -1:
            sex = 'male'
            tag = root[root.find('male\\')+len('male\\'):]
            for f in files:
                ruta = os.path.join(os.getcwd() , os.path.join(root, f))
                data.append({'type': sex, 'tag': tag, 'ruta': ruta})
        if root.find('female\\') > -1:
            sex = 'female'
            tag = root[root.find('female\\')+len('female\\'):]
            for f in files:
                ruta = os.path.join(os.getcwd() , os.path.join(root, f))
                data.append({'type': sex.lower(), 'tag': tag.lower(), 'ruta': ruta})
    return data

def get_savee(data_path="./train_data/not_shared/AudioEmotion/SAVEE"):
    data = []
    for root, dirs, files in os.walk(data_path):
        splited = root[root.find('\\')+len('\\'):].split("_")
        if len(splited) < 2:
            continue
        for f in files:
            ruta = os.path.join(os.getcwd() , os.path.join(root, f))
            data.append({'type': splited[0].lower(), 'tag': splited[1].lower(), 'ruta': ruta})
    return data


def get_ravdess(data_path="./train_data/not_shared/AudioEmotion/RAVDESS"):
    data = []
    for root, dirs, files in os.walk(data_path):
        splited = root[root.find('\\')+len('\\'):].split("_")
        if len(splited) < 2:
            continue
        for f in files:
            ruta = os.path.join(os.getcwd() , os.path.join(root, f))
            data.append({'type': splited[0].lower(), 'tag': splited[1].lower(), 'ruta': ruta})
    return data


def get_tess(data_path="./train_data/not_shared/AudioEmotion/TESS"):
    data = []
    for root, dirs, files in os.walk(data_path):
        splited = root[root.find('\\')+len('\\'):].split("_")
        if len(splited) < 2:
            continue
        for f in files:
            ruta = os.path.join(os.getcwd() , os.path.join(root, f))
            data.append({'type': splited[0].lower(), 'tag': splited[1].lower(), 'ruta': ruta})
    return data