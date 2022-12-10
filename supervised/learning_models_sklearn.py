from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import neural_network


def train_svm(xs: list[list[float]], ys: list, kernel, c, degree, gamma):
	classifier = svm.SVC(kernel=kernel, C=c, degree=degree, gamma=gamma)
	model = classifier.fit(xs, ys)
	return model, classifier


def train_svm_kernel(inputs, outputs, kernel="rbf"):
	classifier = svm.SVC(kernel=kernel)
	model = classifier.fit(inputs, outputs)
	return model, classifier


def train_svm_linear(inputs, outputs, C=1.0):
	classifier = svm.LinearSVC(C=C)
	model = classifier.fit(inputs, outputs)
	return model, classifier


def classify(model, x):
	return model.predict([x])[0]


def train_mlp(inputs, outputs):
	classifier = neural_network.MLPClassifier()
	model = classifier.fit(inputs, outputs)
	return model, classifier


def get_k_scores(clf, inputs: list[list[float]], outputs: list, k: int = 10):
	scores = cross_val_score(clf, inputs, outputs, cv=k)
	return scores
