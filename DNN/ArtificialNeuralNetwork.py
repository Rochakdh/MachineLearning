import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, PReLU, ELU, Dropout

# dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13]  # independent features
y = dataset.iloc[:, 13]  # dependent features

# Create dummy variables
geography = pd.get_dummies(x["Geography"], drop_first=True)
gender = pd.get_dummies(x['Gender'], drop_first=True)

# Concatenate the Data Frames
x = pd.concat([x, geography, gender], axis=1)

# Droping Unnecessary columns
x = x.drop(['Geography', 'Gender'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = Sequential()
classifier.add(Dense(6, input_shape=(11,), kernel_initializer='glorot_uniform', activation='relu'))
classifier.add(Dense(6, kernel_initializer='glorot_uniform', activation='relu'))
classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
classifier.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
model_history = classifier.fit(x_train, y_train, validation_split=0.33, batch_size=10, epochs=100)
