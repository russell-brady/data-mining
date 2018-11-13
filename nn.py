# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = pd.read_csv('Datasets/relevant_data/forBayes2.csv', )
# split into input (X) and output (Y) variables
X = dataset.iloc[:,1:7]
Y = dataset.iloc[:,7]



encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2)

print(X, Y)

# create model
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))