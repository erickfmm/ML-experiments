#True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.
#True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no.
#False Positives (FP) – When actual class is no and predicted class is yes.
#False Negatives (FN) – When actual class is yes but predicted class in no.

#Precision = TP/TP+FP
#Recall = TP/TP+FN

#F1 Score = 2*(Recall * Precision) / (Recall + Precision)
#F1 is usually more useful than accuracy, especially if for an uneven class distribution.


def TruePositives(yreal: list, ypred: list):
    #hese are the correctly predicted
    #positive values which means that
    #the value of actual class is yes
    #and the value of predicted class is also yes.
    tp = 0
    for i in range(len(yreal)):
        if yreal[i] == 1 and ypred[i] == 1:
            tp +=1
    return tp

def TrueNegatives(yreal: list, ypred: list):
    #These are the correctly predicted
    #negative values which means that the
    #value of actual class is no and value
    #of predicted class is also no.
    tn = 0
    for i in range(len(yreal)):
        if yreal[i] == 0 and ypred[i] == 0:
            tn +=1
    return tn

def FalsePositives(yreal: list, ypred: list):
    #When actual class is no and predicted class is yes.
    fp = 0
    for i in range(len(yreal)):
        if yreal[i] == 0 and ypred[i] == 1:
            fp +=1
    return fp

def FalseNegatives(yreal: list, ypred: list):
    #When actual class is yes but predicted class in no.
    fn = 0
    for i in range(len(yreal)):
        if yreal[i] == 1 and ypred[i] == 0:
            fn +=1
    return fn

def Precision(yreal: list, ypred: list):
    TP = TruePositives(yreal, ypred)
    FP = FalsePositives(yreal, ypred)
    return TP/float(TP+FP)

def Recall(yreal: list, ypred: list):
    TP = TruePositives(yreal, ypred)
    FN = FalseNegatives(yreal, ypred)
    return TP/float(TP+FN)

def F1_Score(yreal: list, ypred: list):
    #F1 is usually more useful than accuracy,
    #especially if for an uneven class distribution.
    Recall_ = Recall(yreal, ypred)
    Precision_ = Precision(yreal, ypred)
    return 2.0*(Recall_ * Precision_) / float(Recall_ + Precision_)

def Accuracy(yreal: list, ypred: list):
    count_errors = 0
    for i in range(len(yreal)):
        if yreal[i] != ypred[i]:
            count_errors +=1
    return count_errors/float(len(yreal))