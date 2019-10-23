# ## Step 1.

# ### Importing Library

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ### Loading Data

# load dataset
dataframe = pandas.read_csv("C:/Users/FC/Documents/Deep_Learning_Project_Three/housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]


# ## Step 2. Develop a Baseline Neural Network Model


def baseline_model():
    # create model, write code below
    model = Sequential()
    model.add(Dense(13, activation='relu', input_shape=(13,)))
    model.add(Dense(1))
    # Compile model, write code below
    model.compile(optimizer='Adam',loss='mse', metrics=['mae'])
    return model
    # Compile model
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)



kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ## Step 3: Modeling The Standardized Dataset


# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std

# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ## Step 4.1. Evaluate a Deeper Network Topology

def larger_model():
    # create model, write code below
    model = Sequential()
    model.add(Dense(13, activation='relu', input_shape=(13,)))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))
    # Compile model, write code below
    model.compile(optimizer='Adam',loss='mse', metrics=['mae'])
    return model
    # Compile model
    return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ## Step 4.2. Evaluate a Wider Network Topology

def wider_model():
    # create model, write code below
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(13,)))
    model.add(Dense(1))
    # Compile model, write code below
    model.compile(optimizer='Adam',loss='mse', metrics=['mae'])
    return model
    # Compile model
    return model



numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ## Step 5. Really Scaling up: developing a model that overfits

def overfit_model():
    # create model, write code below
    model = Sequential()
    model.add(Dense(56, activation='relu', input_shape=(13,)))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(74, activation='relu'))
    model.add(Dense(37, activation='relu'))
    model.add(Dense(45, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    # Compile model, write code below
    model.compile(optimizer='Adam',loss='mse', metrics=['mae'])
    return model
    # Compile model
    return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=overfit_model, epochs=200, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("over-fit: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ## Step 6. Tuning the Model
def tuned_model():
    # create model, write code below
    model = Sequential()
    model.add(Dense(13, activation='relu', input_shape=(13,)))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))
    # Compile model, write code below
    model.compile(optimizer='Adam',loss='mse', metrics=['mae'])
    return model
    # Compile model
    return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=tuned_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("tuned: %.2f (%.2f) MSE" % (abs(results.mean()), results.std()))


# ## Step 7. Rewriting the code using the Keras Functional API

from keras.models import Model
from keras.layers import Input , Dense
def functional_model():
    inputs= Input(shape=(13,))
    x = Dense(13 , activation='relu')(inputs)
    x1= Dense(8 , activation='relu')(x)
    outputs= Dense(1)(x1)
    model = Model(inputs, outputs)
    model.compile(optimizer= 'Adam' , loss='mse' , metrics=['mae'])
    return model
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=functional_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("tuned: %.2f (%.2f) MSE" % (abs(results.mean()), results.std()))


# ## Step 8. Rewriting the code by doing Model Subclassing
import tensorflow as tf
class Mymodel(tf.keras.Model):
    def __init__(self):
        super(Subclass , self).__init__()
        self.dense1 = tf.keras.layers.Dense(13 , activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(6 , activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(1)
    def call(self , inputs):
        a=self.dense1(inputs)
        x= self.dense2(a)
        return self.dense3(x)
def subclass_model():
    model= Subclass()
    model.compile(optimizer='Adam' , loss='mse' , metrics=['mae'])
    return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize' , StandardScaler()))
estimators.append(('mlp' , KerasRegressor(build_fn = subclass_model , epochs = 50 , batch_size = 5 , verbose = 0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits = 10 , random_state = seed)
results = cross_val_score(pipeline , X , Y , cv = kfold)
print("Results: %2f(%2f)MSE"%(abs(results.mean()),results.std()))


# ## Step 9. Rewriting the code without using scikit-learn
data1 = pandas.read_csv("C:/Users/FC/Documents/Deep_Learning_Project_Three/housing.csv", delim_whitespace=True, header=None)
data = data1.values
train_data = data[:404,0:13]
train_labels = data[:404,13]
test_data = data[404:,0:13]
test_labels = data[404:,13]

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

from keras import models
from keras import layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

import numpy as np
k=4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate( [train_labels[:i * num_val_samples], train_labels[(i + 1) * num_val_samples:]],axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

all_scores

np.mean(all_scores)

model = build_model()
model.fit(train_data, train_labels,
epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_labels)
print(test_mse_score)
