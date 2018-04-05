import neuralnetworks.perceptron_random as pr


Xs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
Ys = [0, 0, 0, 1]

def train_perceptron():
    print("creating perceptron")
    p = pr.PerceptronRandom()
    print("to train")
    print(p.train(Xs, Ys, 0.7, 100, -1, 1, 0.05))
    print(p.weights)

train_perceptron()