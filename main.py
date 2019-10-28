import sklearn.utils as utils
import sklearn.datasets as datasets
from knn import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
ynames = iris.target_names

X, y = utils.shuffle(X, y, random_state=1)
train_set_size = 100
X_train = X[:train_set_size]  # selects first 100 rows (examples) for train set
y_train = y[:train_set_size]
X_test = X[train_set_size:]   # selects from row 100 until the last one for test set
y_test = y[train_set_size:]

k = 5
knn = KNeighborsClassifier(k=k)

knn.fit(X_train, y_train)
y_pred_test = knn.predict(X_test)
print("Accuracy of KNN test set:", knn.score(y_pred_test, y_test))