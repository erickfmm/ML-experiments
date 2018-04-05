from sklearn.datasets import load_iris
import neuralnetworks.perceptron_random as pr

iris = load_iris()

#print(iris.data)
#print(iris.target)

def train_perceptron():
    print("creating perceptron")
    p = pr.PerceptronRandom()
    print("to train")
    print(p.train(iris.data[:100], iris.target[:100], 1.0, 4, -1, 1, 0.05))
    print(p.weights)

train_perceptron()