import csv
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import preprocessing.change_types as change_types
from os.path import join

__all__ = ["LoadTitanic",]

class LoadTitanic(ILoadSupervised):
    def __init__(self, folderPath="train_data/Folder_Basic/titanic/"):
        self.TYPE = SupervisedType.Classification
        self.folderPath = folderPath
        #_, self.XTest, _, self.headers = self.read_titanicfile(join(folderPath, "test.csv"))
        #self.classes = ["Not survived", "Survived"]
    
    def get_all(self):
        Xs = []
        Ys = []
        for x, y in self.read_titanicfile(join(self.folderPath, "train.csv")):
            Xs.append(x)
            Ys.append(y)
        return Xs, Ys
    
    def get_classes(self):
        return ["Not survived", "Survived"] #[1, 0]
    
    def get_headers(self):
        return ["Sex", "Pclass", "Age", "SibSp", "Fare", "Embarked"]
    
    def get_all_yielded(self):
        for x, y in self.read_titanicfile(join(self.folderPath, "train.csv")):
            yield x, y

    @staticmethod
    def read_titanicfile(filepath):
        with open(filepath, 'r') as t_file:
            t_csv = csv.DictReader(t_file)
            for row in t_csv:
                #passengerId = int(row['PassengerId'])
                sex = 0 if row['Sex'] == 'male' else 1
                if row['Embarked'] == '':
                    embarked = 0
                elif row['Embarked'] == 'S':
                    embarked = 1
                elif row['Embarked'] == 'C':
                    embarked = 2
                elif row['Embarked'] == 'Q':
                    embarked = 3
                if 'Survived' in row:
                    y = int(row['Survived'])
                x = [
                    sex,
                    int(row['Pclass']),
                    change_types.to_int(row['Age'], -1),
                    int(row['SibSp']),
                    int(row['Parch']),
                    change_types.to_float(row['Fare'], 0.0),
                    embarked
                    ]
                yield x, y
