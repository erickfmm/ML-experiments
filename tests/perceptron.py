import with_nolib.supervised.perceptron as pr

def train(Xs, Ys):
    print("creating perceptron")
    p = pr.Perceptron()
    print("to train")
    print(p.train(Xs, Ys, 0.7, 100, -1, 1, 0.05))
    print(p.weights)