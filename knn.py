import dataset as ds
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import imutils

(X_train, X_test, y_train, y_test) = train_test_split(
	X, y, test_size=0.15, random_state=42)

model = KNeighborsClassifier(n_neighbors=args["neighbors"])
model.fit(X_train,y_train)
acc = model.score(X_test, y_test)
print("k-NN classifier: k=%d" % args["neighbors"])
print("Accuracy: {:.2f}%".format(acc * 100))