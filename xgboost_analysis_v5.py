import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures
from sklearn.decomposition import PCA, FastICA,SparsePCA,KernelPCA
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from bayes_opt import BayesianOptimization


def xgb_r2_score(preds, dtrain):
    # Courtesy of Tilii
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

def train_xgb(max_depth, subsample, min_child_weight, gamma, colsample_bytree):
    # Evaluate an XGBoost model using given params
    xgb_params = {
        'n_trees': 550,
        'eta': 0.008,
        'max_depth': int(max_depth),
        'subsample': max(min(subsample, 1), 0),
        'objective': 'reg:linear',
        'base_score': y_mean, # base prediction = mean(target)
        'silent': 1,
        'min_child_weight': int(min_child_weight),
        'gamma': max(gamma, 0),
        'colsample_bytree': max(min(colsample_bytree, 1), 0)
    }
    scores = xgb.cv(xgb_params, dtrain, num_boost_round=1500, early_stopping_rounds=50, verbose_eval=False, feval=xgb_r2_score, maximize=True, nfold=5)['test-r2-mean'].iloc[-1]
    return scores

base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
train = pd.read_csv(base_input_path+'train.csv')
test = pd.read_csv(base_input_path+'test.csv')


# Remove the outlier
train=train[train.y<250]


# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))


# 2.identify constant features by looking at the standard deviation (check id std ==0.0)
#commented below code it does not have much impact on r2 score

# print("identiying constant columns")
# desc = train.describe().transpose()
# columns_to_drop = desc.loc[desc["std"] == 0].index.values
# print ('constants columns to drop in train {}'.format(columns_to_drop))
#
# print ("Dropping constant columns from train")
# train.drop(columns_to_drop, axis=1, inplace=True)
#test.drop(columns_to_drop, axis=1, inplace=True)



# Remove the outlier
train=train[train.y<250]

#
# # Check no. of rows greater than equal to 100
# print ("number of rows greater than equal to 100 - {}".format(len(train['y'][(train.y>=100)])))
#
# # Check no. of rows less than 100
# print ("number of rows less than 100 - {}".format(len(train['y'][(train.y<100)])))
#
#
# #Now we convert the training set into a classification problem. Create a new field for class.
#
# train['y_class'] = train.y.apply(lambda x: 0 if x<100  else 1 )
# test['y_class']= train['y_class']
# # # Concat the datasets
# # data = pd.concat([train,test])
#




# shape
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

##Add decomposed components: PCA / ICA etc.


n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# # PCA
# pca = PCA(n_components=n_comp, random_state=420)
# pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
# pca2_results_test = pca.transform(test)
#

#Polynomial features
# poly = PolynomialFeatures(degree=1)
# poly_results_train = poly.fit_transform(train.drop(["y"], axis=1))
# poly_results_test = poly.transform(test)

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



# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    # train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    # test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    # train['poly_' + str(i)] = poly_results_train[:, i - 1]
    # test['poly_' + str(i)] = poly_results_test[:, i - 1]

    train['spca_' + str(i)] = spca2_results_train[:, i - 1]
    test['spca_' + str(i)] = spca2_results_test[:, i - 1]

    train['kpca_' + str(i)] = kpca2_results_train[:, i - 1]
    test['kpca_' + str(i)] = kpca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]




y_train = train["y"]
y_mean = np.mean(y_train)

# A parameter grid for XGBoost
params = {
  'min_child_weight':(1, 20),
  'gamma':(0, 10),
  'subsample':(0.5, 1),
  'colsample_bytree':(0.1, 1),
  'max_depth': (2, 15)
}

# Initialize BO optimizer
xgb_bayesopt = BayesianOptimization(train_xgb, params)



# prepare dict of params for xgboost to run with
# xgb_params = {
#     'n_trees': 500,
#     'eta': 0.005,
#     'max_depth': 4,
#     'subsample': 0.921,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     'base_score': y_mean,  # base prediction = mean(target)
#     'silent': 1
# }

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
# cv_result = xgb.cv(xgb_params,
#                   dtrain,
#                   num_boost_round=1000, # increase to have better results (~700)
#                   early_stopping_rounds=50,
#                   verbose_eval=10,
#                   show_stdv=False
#                  )
#
# num_boost_rounds = len(cv_result)
# print('num_boost_rounds=' + str(num_boost_rounds))

# Maximize R2 score
xgb_bayesopt.maximize(init_points=5, n_iter=25)

# Get the best params
p = xgb_bayesopt.res['max']['max_params']

xgb_params = {
    'n_trees': 550,
    'eta': 0.008,
    'max_depth': int(p['max_depth']),
    'subsample': max(min(p['subsample'], 1), 0),
    'objective': 'reg:linear',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1,
    'min_child_weight': int(p['min_child_weight']),
    'gamma': max(p['gamma'], 0),
    'colsample_bytree': max(min(p['colsample_bytree'], 1), 0)
}

model = xgb.train(xgb_params, dtrain, num_boost_round=1500, verbose_eval=False, feval=xgb_r2_score, maximize=True)

# num_boost_rounds = 1400
# train model
# model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# check f2-score (to get higher score - increase num_boost_round in previous cell)

r2_score =r2_score(dtrain.get_label(), model.predict(dtrain))

print(r2_score)

# make predictions and save results
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv(base_output_path+'xgboost-v5-r2-{0}-pca-ica.csv'.format(r2_score), index=False)