from keras.datasets import boston_housing
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import numpy as np

__all__ = ["LoadBostonHousing",]

class LoadBostonHousing(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Regression
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
        return None
    
    def get_headers(self):
        return self.headers

    def get_splited(self, test_split=0.2, seed = 113):
        (XTrain,YTrain),(XTest,YTest)=boston_housing.load_data(test_split=test_split, seed=seed)
        return (XTrain, YTrain), (XTest, YTest)
    
    def get_all(self):
        (XTrain,YTrain),(_, _)=boston_housing.load_data(test_split=0)
        return XTrain,YTrain
    
    def get_all_yielded(self):
        xs, ys = self.get_all()
        for i in range(len(xs)):
            yield xs[i], ys[i]
    
    def get_train_yielded(self, test_split=0.2, seed = 113):
        XTrain, YTrain = self.get_train(test_split=test_split, seed=seed)
        for i in range(len(XTrain)):
            yield XTrain[i], YTrain[i]

    def get_test_yielded(self, test_split=0.2, seed = 113):
        XTest, YTest = self.get_test(test_split=test_split, seed=seed)
        for i in range(len(XTest)):
            yield XTest[i], YTest[i]

    def get_train(self, test_split=0.2, seed = 113):
        (XTrain, YTrain), (_, _) = self.get_splited(test_split=test_split, seed=seed)
        return XTrain, YTrain

    def get_test(self, test_split=0.2, seed = 113):
        (_, _), (XTest, YTest) = self.get_splited(test_split=test_split, seed=seed)
        return XTest, YTest