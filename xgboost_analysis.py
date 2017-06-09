import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, FastICA
import xgboost as xgb
from sklearn.metrics import r2_score

base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
train = pd.read_csv(base_input_path+'train.csv')
test = pd.read_csv(base_input_path+'test.csv')

test_ids = test['ID']

# identify constant features by looking at the standard deviation (check id std ==0.0)
desc_train = train.describe().transpose()
columns_to_drop = desc_train.loc[desc_train["std"] == 0].index.values
train.drop(columns_to_drop, axis=1, inplace=True)

# check which column has been dropped
print(columns_to_drop)

desc_test = test.describe().transpose()
columns_to_drop = desc_test.loc[desc_test["std"] == 0].index.values
train.drop(columns_to_drop, axis=1, inplace=True)

# check which column has been dropped
print(columns_to_drop)

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# # One-hot encoding of categorical/strings
# train = pd.get_dummies(train, drop_first=True)
# test = pd.get_dummies(test, drop_first=True)

# shape
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
y_train = train["y"]
y_mean = np.mean(y_train)


#PCA/ICA for dimensionality reduction
n_comp = 10

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

train_cols = [col for col in list(train)]
test_cols = [col for col in list(test)]

print (train_cols)
print (test_cols)

train.drop(train_cols, axis=1, inplace=True)
test.drop(test_cols, axis=1, inplace=True)

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]


print("After PCA/ICA")
print (len(list(train)))
print (len(list(test)))

# print("Deleting old columns and only retaining PCA/ICA columns")
#
# filter_pca_col_train = [col for col in list(train) if  col.startswith('pca_') ] # or not col.startswith('ica_')]
# filter_pca_col_test = [col for col in list(test) if  col.startswith('pca_')] # or not col.startswith('ica_')]
#
# filter_ica_col_train = [col for col in list(train) if  col.startswith('ica_') ] # or not col.startswith('ica_')]
# filter_ica_col_test = [col for col in list(test) if  col.startswith('ica_')] # or not col.startswith('ica_')]
#
# x= filter_ica_col_train+filter_ica_col_train
#
# print ("Deleting no PCA/ICA columns")
# print ("train columns {}".format(filter_pca_col_train))
# print ("test columns {}".format(filter_pca_col_test))
#
# # train.drop(filter_pca_col_train, axis=1, inplace=True)
# # test.drop(filter_pca_col_test, axis=1, inplace=True)
#
# train.drop(filter_ica_col_train, axis=1, inplace=True)
# test.drop(filter_ica_col_test, axis=1, inplace=True)
#
#
# print("After Deleting old columns and only retaining PCA/ICA columns")
# print (len(list(train)))
# print (len(list(test)))


#xgboost
# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 500,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1,
    'colsample_bytree':0.4,
    'n_estimators':1300
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   num_boost_round=700, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50,
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)



# check f2-score (to get higher score - increase num_boost_round in previous cell)
# now fixed, correct calculation
r2_score =r2_score(dtrain.get_label(), model.predict(dtrain))

print(r2_score)


# make predictions and save results
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test_ids.astype(np.int32), 'y': y_pred})
output.to_csv(base_output_path+'xgboost-r2-{0}-pca-ica.csv'.format(r2_score), index=False)






