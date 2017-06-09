import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA, FastICA
import xgboost as xgb
from sklearn.metrics import r2_score

pd.options.mode.chained_assignment = None  # default='warn'

base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
train = pd.read_csv(base_input_path + 'train.csv')
test = pd.read_csv(base_input_path + 'test.csv')
train_y = train['y']

# 1.one hot encoding
num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
print ("One hot encoding")
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]

# 2.identify constant features by looking at the standard deviation (check id std ==0.0)
print("identiying constant columns")
desc = train.describe().transpose()
columns_to_drop = desc.loc[desc["std"] == 0].index.values
train.drop(columns_to_drop, axis=1, inplace=True)
print ('constants columns to drop in train {}'.format(columns_to_drop))


#2.1 feature selection
print ("feature selection using random forest regressior")

# Tree-based estimators can be used to compute feature importances,
# which in turn can be used to discard irrelevant features

clf = RandomForestRegressor(n_estimators=100, max_features='log2')
clf = clf.fit(train, train_y)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(5, 60))

# important_features = features[features['importance']>0]
#
# print ("feature importance {}".format(important_features))
#
# train= train[important_features]
# test= test[important_features]

# desc = test.describe().transpose()
# columns_to_drop = desc.loc[desc["std"] == 0].index.values
# test.drop(columns_to_drop, axis=1, inplace=True)
#
# print ('constants columns to drop in test {}'.format(columns_to_drop))

#Dimension reduction
#1.PCA
n_comp = 40

print('dimension reduction')
# PCA
print("PCA")
pca =  PCA() #PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train)
pca2_results_test = pca.transform(test)
# X_rec = pca.inverse_transform(train)
# print (X_rec)
