import csv
from load_data.ILoadSupervised import ILoadSupervised
import utils

__all__ = ["LoadTitanic",]

class LoadTitanic(ILoadSupervised):
    def __init__(self):
        _, self.XTrain, self.YTrain = self.read_titanicfile("train_data/shared/titanic/train.csv")
        _, self.XTest, _ = self.read_titanicfile("train_data/shared/titanic/test.csv")

    def get_default(self):
        return self.XTrain, self.YTrain

    def get_splited(self):
        return None
    
    def get_all(self):
        return self.XTrain, self.YTrain
    
    @staticmethod
    def read_titanicfile(filepath):
        titanic = []
        features = []
        targets = []
        with open(filepath, 'r') as t_file:
            t_csv = csv.DictReader(t_file)
            for row in t_csv:
                titanic.append(row)
            for j in range(len(titanic)):
                titanic[j]['PassengerId'] = int(titanic[j]['PassengerId'])
                titanic[j]['Sex'] = 0 if titanic[j]['Sex'] == 'male' else 1
                titanic[j]['Pclass'] = int(titanic[j]['Pclass'])
                titanic[j]['Age'] = utils.change_types.to_int(titanic[j]['Age'], -1)
                titanic[j]['SibSp'] = int(titanic[j]['SibSp'])
                titanic[j]['Parch'] = int(titanic[j]['Parch'])
                titanic[j]['Fare'] = utils.change_types.to_float(titanic[j]['Fare'], 0.0)
                if titanic[j]['Embarked'] == '':
                    titanic[j]['Embarked'] = 0
                elif titanic[j]['Embarked'] == 'S':
                    titanic[j]['Embarked'] = 1
                elif titanic[j]['Embarked'] == 'C':
                    titanic[j]['Embarked'] = 2
                elif titanic[j]['Embarked'] == 'Q':
                    titanic[j]['Embarked'] = 3
                if 'Survived' in titanic[j]:
                    targets.append(int(titanic[j]['Survived']))
                features.append([
                    titanic[j]['Sex'], 
                    titanic[j]['Pclass'], 
                    titanic[j]['Age'], 
                    titanic[j]['SibSp'], 
                    titanic[j]['Parch'], 
                    titanic[j]['Fare'], 
                    titanic[j]['Embarked']
                    ])
        return titanic, features, targets
