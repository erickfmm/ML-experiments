from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import neural_network

def train_svm(inputs, outputs, kernel, C, degree, gamma):
	clf = svm.SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
	model = clf.fit(inputs, outputs)
	return model, clf

def train_svm_kernel(inputs, outputs, kernel="rbf"):
	clf = svm.SVC(kernel=kernel)
	model = clf.fit(inputs, outputs)
	return model, clf

def train_svm_linear(inputs, outputs, C=1.0):
	clf = svm.LinearSVC(C=C)
	model = clf.fit(inputs, outputs)
	return model, clf

def classify(model, input):
	return model.predict([input])[0]

def train_MLP(inputs, outputs):
	clf = neural_network.MLPClassifier()
	model = clf.fit(inputs, outputs)
	return model, clf

def get_k_scores(clf, inputs, outputs, k=10):
	scores = cross_val_score(clf, inputs, outputs, cv=k)
	return scores