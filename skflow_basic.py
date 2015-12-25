''' simple skflow examples
original source = https://github.com/google/skflow
'''
import skflow
from sklearn import datasets, metrics, preprocessing


# Simple linear classification:

iris = datasets.load_iris()
classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(classifier.predict(iris.data), iris.target)
print("Accuracy: %f" % score)

# Simple linear regression:

boston = datasets.load_boston()
X = preprocessing.StandardScaler().fit_transform(boston.data)
regressor = skflow.TensorFlowLinearRegressor()
regressor.fit(X, boston.target)
score = metrics.mean_squared_error(regressor.predict(X), boston.target)
print ("MSE: %f" % score)