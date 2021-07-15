import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv("car.data")
print(data.head())

# converting data to labels
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
print(x_train, x_test)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
pridected = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(pridected)):
    print("Data:", x_test[x], "Actual:", names[y_test[x]], "Predicted:", names[pridected[x]])
