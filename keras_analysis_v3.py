# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math as mt
from pylab import savefig

# preprocessing/decomposition
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

# Keras is a deep learning library that wraps the efficient numerical libraries Theano and TensorFlow.
# It provides a clean and simple API that allows you to define and evaluate deep learning models in just a few lines of code.from keras.models import Sequential, load_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import LSTM
# define custom R2 metrics for Keras backend
from keras import backend as K
# to tune the NN
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam

# model evaluation
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA,SparsePCA,KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNetCV, LassoLarsCV, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline, make_union

class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

# feature selection
from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold

# define path to save model
import os

# to make results reproducible
seed = 42

# Read datasets
base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
train = pd.read_csv(base_input_path+'train.csv')
test = pd.read_csv(base_input_path+'test.csv')

model_path = base_output_path+'keras_model.h5'

# Remove the outlier
# train=train[train.y<250]

# save IDs for submission
id_test = test['ID'].copy()

###########################
# DATA PREPARATION
###########################

# glue datasets together
total = pd.concat([train, test], axis=0)
print('initial shape: {}'.format(total.shape))

# binary indexes for train/test set split
is_train = ~total.y.isnull()

# find all categorical features
cf = total.select_dtypes(include=['object']).columns

# make one-hot-encoding convenient way - pandas.get_dummies(df) function
dummies = pd.get_dummies(
    total[cf],
    drop_first=False  # you can set it = True to ommit multicollinearity (crucial for linear models)
)

print('oh-encoded shape: {}'.format(dummies.shape))

# get rid of old columns and append them encoded
total = pd.concat(
    [
        total.drop(cf, axis=1),  # drop old
        dummies  # append them one-hot-encoded
    ],
    axis=1  # column-wise
)

print('appended-encoded shape: {}'.format(total.shape))

# recreate train/test again, now with dropped ID column
train, test = total[is_train].drop(['ID'], axis=1), total[~is_train].drop(['ID', 'y'], axis=1)

# drop redundant objects
del total

# check shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))


n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
# pca = PCA(n_components=n_comp, random_state=420)
# pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
# pca2_results_test = pca.transform(test)

#sparse PCA
spca = SparsePCA(n_components=n_comp, random_state=420)
spca2_results_train = spca.fit_transform(train.drop(["y"], axis=1))
spca2_results_test = spca.transform(test)

#Kernel PCA
kpca = KernelPCA(n_components=n_comp, random_state=420)
kpca2_results_train = kpca.fit_transform(train.drop(["y"], axis=1))
kpca2_results_test = kpca.transform(test)


# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

# save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    # train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    # test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['spca_' + str(i)] = spca2_results_train[:, i - 1]
    test['spca_' + str(i)] = spca2_results_test[:, i - 1]

    train['kpca_' + str(i)] = kpca2_results_train[:, i - 1]
    test['kpca_' + str(i)] = kpca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]


y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values

#########################################################################################################################################
# GENERATE MODEL
# The Keras wrappers require a function as an argument.
# This function that we must define is responsible for creating the neural network model to be evaluated.
# Below we define the function to create the baseline model to be evaluated.
# The network uses good practices such as the rectifier activation function for the hidden layer.
# No activation function is used for the output layer because it is a regression problem and we are interested in predicting numerical
# values directly without transform.# The efficient ADAM optimization algorithm is used and a mean squared error loss function is optimized.
# This will be the same metric that we will use to evaluate the performance of the model.
# It is a desirable metric because by taking the square root gives us an error value we can directly understand in the context of the problem.
##########################################################################################################################################

def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# Base model architecture definition.
# Dropout is a technique where randomly selected neurons are ignored during training.
# They are dropped-out randomly. This means that their contribution to the activation.
# of downstream neurons is temporally removed on the forward pass and any weight updates are
# not applied to the neuron on the backward pass.
# More info on Dropout here http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# BatchNormalization, Normalize the activations of the previous layer at each batch, i.e. applies a transformation
# that maintains the mean activation close to 0 and the activation standard deviation close to 1.
def model():
    model = Sequential()
    # Input layer with dimension input_dims and hidden layer i with input_dims neurons.
    model.add(Dense(input_dims, input_dim=input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims // 2, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Output Layer.
    model.add(Dense(1))
    # Use a large learning rate with decay and a large momentum.
    # Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # compile this model
    model.compile(loss='mean_squared_error',  # one may use 'mean_absolute_error' as alternative
                  optimizer='rmsprop',
                  metrics=[r2_keras, "mse"]  # you can add several if needed
                  )

    # Visualize NN architecture
    print(model.summary())
    return model


# initialize input dimension

input_dims = train.shape[1] - 1
# input_dims = train_reduced.shape[1]

# make np.seed fixed
np.random.seed(seed)

# initialize estimator, wrap model in KerasRegressor
estimator = KerasRegressor(
    build_fn=model,
    nb_epoch=300,
    batch_size=30,
    verbose=1
)

# X, y preparation
X, y = train.drop('y', axis=1).copy().values, train.y.values
X_test = test.values
print('\nTrain shape No Feature Selection: {}\nTest shape No Feature Selection: {}'.format(X.shape, X_test.shape))

###############
# K-FOLD
###############
"""Return the sample arithmetic mean of data."""


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


"""Return sum of square deviations of sequence data."""


def sum_of_square_deviation(numbers, mean):
    return float(1 / len(numbers) * sum((x - mean) ** 2 for x in numbers))


'''    
n_splits = 2
kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
kf.get_n_splits(X)

mse_scores = list()
r2_scores = list()

for fold, (train_index, test_index) in enumerate(kf.split(X)):

    print("TRAIN:", train_index, "TEST:", test_index)
    X_tr, X_val = X[train_index], X[test_index]
    y_tr, y_val = y[train_index], y[test_index]

    # prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_r2_keras', 
            patience=20,
            mode='max',
            verbose=1)
    ]
    # fit estimator
    history = estimator.fit(
        X_tr, 
        y_tr, 
        epochs=500,
        validation_data=(X_val, y_val),
        verbose=2,
        callbacks=callbacks,
        shuffle=True
    )

    pred = estimator.predict(X_val)

    mse = mean_squared_error(y_val, estimator.predict(X_val))**0.5
    r2 = r2_score(y_val, estimator.predict(X_val))
    mse_scores.append(mse)
    r2_scores.append(r2)

    print('Fold %d: Mean Squared Error %f'%(fold, mse))
    print('Fold %d: R^2 %f'%(fold, r2))

mean_mse = mean(mse_scores)
mean_r2 = mean(r2_scores)

standard_deviation_mse = mt.sqrt(sum_of_square_deviation(mse_scores,mean_mse))
standard_deviation_r2 = mt.sqrt(sum_of_square_deviation(r2_scores,mean_r2))

print('=====================')
print( 'Mean Squared Error %f'%mean_mse)
print('=====================')
print('=====================')
print( 'Stdev Squared Error %f'%standard_deviation_mse)
print('=====================')
print('=====================')
print( 'Mean R^2 %f'%mean_r2)
print('=====================')
print('=====================')
print( 'Stdev R^2 %f'%standard_deviation_r2)
print('=====================')
'''
# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_r2_keras',
        patience=20,
        mode='max',
        verbose=1),
    ModelCheckpoint(
        model_path,
        monitor='val_r2_keras',
        save_best_only=True,
        mode='max',
        verbose=0)
]

# train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=seed
)

# fit estimator
history = estimator.fit(
    X_tr,
    y_tr,
    epochs=500,
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=callbacks,
    shuffle=True
)

# # list all data in history
# print(history.history.keys())
#
# # summarize history for R^2
# fig_acc = plt.figure(figsize=(10, 10))
# plt.plot(history.history['r2_keras'])
# plt.plot(history.history['val_r2_keras'])
# plt.title('model accuracy')
# plt.ylabel('R^2')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# fig_acc.savefig(base_output_path+"model_accuracy.png")
#
# # summarize history for loss
# fig_loss = plt.figure(figsize=(10, 10))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# fig_loss.savefig(base_output_path+"model_loss.png")
#
# # if best iteration's model was saved then load and use it
# if os.path.isfile(model_path):
#     estimator = load_model(model_path, custom_objects={'r2_keras': r2_keras})
#
# # Plot in blue color the predicted data and in green color the
# # actual data to verify visually the accuracy of the model.
# predicted = estimator.predict(X_val)
# fig_verify = plt.figure(figsize=(100, 50))
# plt.plot(predicted, color="blue")
# plt.plot(y_val, color="green")
# plt.title('prediction')
# plt.ylabel('value')
# plt.xlabel('row')
# plt.legend(['predicted', 'actual data'], loc='upper left')
# plt.show()
# fig_verify.savefig(base_output_path+"model_verify.png")

# finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays)
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

'''Train the stacked models then predict the test data'''

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25, max_depth=3)),
    StackingEstimator(estimator=ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35, max_features=150)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(
        estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55,
                                            min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()

)

stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

# check performance on train set
print('MSE train: {}'.format(mean_squared_error(y_tr, estimator.predict(X_tr)) ** 0.5))  # mse train
print('R^2 train: {}'.format(r2_score(y_tr, estimator.predict(X_tr))))  # R^2 train

# check performance on validation set
print('MSE val: {}'.format(mean_squared_error(y_val, estimator.predict(X_val)) ** 0.5))  # mse val

r2_val= r2_score(y_val, estimator.predict(X_val))

print('R^2 val: {}'.format(r2_val) ) # R^2 val
pass

# predict results
res = estimator.predict(X_test).ravel()
print(res)

# create df and convert it to csv
output = pd.DataFrame({'id': id_test, 'y': res})
output.to_csv(base_output_path+'keras-analysis0v3-r2-{}.csv'.format(r2_val), index=False)