from keras.datasets import boston_housing
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import numpy as np

__all__ = ["LoadBostonHousing",]

class LoadBostonHousing(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        (self.XTrain,self.YTrain),(self.XTest,self.YTest)=boston_housing.load_data()
        self.headers = ["CRIM     per capita crime rate by town",
            "ZN       proportion of residential land zoned for lots over 25,000 sq.ft.",
            "INDUS    proportion of non-retail business acres per town",
            "CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
            "NOX      nitric oxides concentration (parts per 10 million)",
            "RM       average number of rooms per dwelling",
            "AGE      proportion of owner-occupied units built prior to 1940",
            "DIS      weighted distances to five Boston employment centres",
            "RAD      index of accessibility to radial highways",
            "TAX      full-value property-tax rate per $10,000",
            "PTRATIO  pupil-teacher ratio by town",
            "B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
            "LSTAT    % lower status of the population",
            "MEDV     Median value of owner-occupied homes in $1000's"]
    
    def get_classes(self):
        return max([max(self.YTrain), max(self.YTest)]) - min([min(self.YTrain), min(self.YTest)])
    
    def get_headers(self):
        return self.headers

    def get_default(self):
        return self.XTrain, self.YTrain

    def get_splited(self):
        return (self.XTrain, self.YTrain), (self.XTest, self.YTest)
    
    def get_all(self):
        return np.append(self.XTrain, self.XTest, 0), np.append(self.YTrain, self.YTest, 0)