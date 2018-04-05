import csv
import neuralnetworks.perceptron_random as pr

titanic = []
features = []
target = []

def to_int(number, default, base=10, throw_error=False):
    x = default
    try:
        x = int(number, base)
    except:
        if throw_error:
            raise ValueError("not int")
            return None
    return x

with open('train_data/titanic.csv', 'r') as t_file:
    t_csv = csv.DictReader(t_file)
    for row in t_csv:
        titanic.append(row)
    for j in range(len(titanic)):
        titanic[j]['PassengerId'] = int(titanic[j]['PassengerId'])
        titanic[j]['Sex'] = 0 if titanic[j]['Sex'] == 'male' else 1
        titanic[j]['Pclass'] = int(titanic[j]['Pclass'])
        titanic[j]['Age'] = to_int(titanic[j]['Age'], -1)
        titanic[j]['SibSp'] = int(titanic[j]['SibSp'])
        titanic[j]['Parch'] = int(titanic[j]['Parch'])
        titanic[j]['Fare'] = float(titanic[j]['Fare'])
        if titanic[j]['Embarked'] == '':
            titanic[j]['Embarked'] = 0
        elif titanic[j]['Embarked'] == 'S':
            titanic[j]['Embarked'] = 1
        elif titanic[j]['Embarked'] == 'C':
            titanic[j]['Embarked'] = 2
        elif titanic[j]['Embarked'] == 'Q':
            titanic[j]['Embarked'] = 3
        target.append(int(titanic[j]['Survived']))
        features.append([
            titanic[j]['Sex'], 
            titanic[j]['Pclass'], 
            titanic[j]['Age'], 
            titanic[j]['SibSp'], 
            titanic[j]['Parch'], 
            titanic[j]['Fare'], 
            titanic[j]['Embarked']
            ])

def train_perceptron():
    print("creating perceptron")
    p = pr.PerceptronRandom()
    print("to train")
    print(p.train(features, target, 0.9, 30, -1, 1, 0.05))
    print(p.weights)

train_perceptron()