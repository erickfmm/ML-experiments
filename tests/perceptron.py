import with_nolib.supervised.perceptron as pr

def train(Xs, Ys):
    print("creating perceptron")
    p = pr.Perceptron()
    print("to train")
    print(p.train(Xs, Ys, threshold=1, maxIterations=100, minValue=-1, maxValue=1, learning_rate=0.05))
    print(p.weights)