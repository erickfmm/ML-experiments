from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import os
import pims

__all__ = ["LoadRecognitionHumanActions",]

class LoadRecognitionHumanActions(ILoadSupervised):
    def __init__(self, folderPath="train_data/not_shared/Recognition of human actions/"):
        self.TYPE = SupervisedType.Classification
        self.folderPath = folderPath
        self.headers = ["pixels"]
        self.classes = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        self.get_frames()
        self.Xs = []
        self.Ys = []
        i_set = 0
        for video_dict in self.video_metadata_dict:
            video_folder_path = os.path.join(self.folderPath, video_dict["tag"])
            video_path = None
            for file_name in os.listdir(video_folder_path):
                if file_name.find(video_dict["name"]) == 0:
                    video_path = os.path.join(video_folder_path, file_name)
                    break
            if video_path is not None: #the video was found
                v = pims.Video(video_path)
                print(video_dict["name"]+": ", len(v))
                print(video_path)
                for frame_set in video_dict["frames"]: #a video has many instances
                    if frame_set[-1] > len(v)-1:
                        print("the video is not so long, probably is corrupted")
                        print("frame set ended in: ", frame_set[-1], " but video has ", len(v), " frames")
                        print(str(i_set)+" - "+video_dict["name"]+": ", len(v))
                        break
                    self.Xs.append([])
                    self.Ys.append([])
                    for i_frame in frame_set:
                        #print(i_frame, end=", ")
                        self.Xs[i_set].append(v[i_frame][:, :, 0]) #slicing because its only black and white
                        self.Ys[i_set].append(video_dict["tag"])
                    i_set += 1
                    #print()
        return self.Xs, self.Ys
    
    def get_frames(self, filePath="00sequences.txt"):
        i = 0
        self.video_metadata_dict = []
        with open(os.path.join(self.folderPath, filePath), "r") as sequences_file:
            #sequences_file = open(os.path.join(self.folderPath, filePath), "r")
            for line in sequences_file:
                i += 1
                if i >= 22:
                    if line.find("person") == 0 and line.find("frames") > 0:
                        name = line[0:line.find("\t")]
                        frame_string = "frames\t"
                        frames = line[line.find(frame_string)+len(frame_string):line.find("\n")]
                        frames = frames.split(", ")
                        fs_limits = []
                        fs = []
                        n_person = name.split("_")[0]
                        n_person = n_person[n_person.find("person")+len("person"):]
                        n_person = int(n_person, 10)
                        for i_frame_set in range(len(frames)):
                            sp = frames[i_frame_set].split("-")
                            start_frame = int(sp[0], 10)-1
                            end_frame = int(sp[1], 10)-1
                            fs_limits.append((start_frame, end_frame))
                            fs.append([])
                            for i_frame in range(start_frame, end_frame+1):
                                fs[i_frame_set].append(i_frame)
                        self.video_metadata_dict.append({
                            "name": name,
                            "n_person": n_person,
                            "tag":name.split("_")[1] ,
                            "frame_limits": fs_limits,
                            "frames": fs
                            })
