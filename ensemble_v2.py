import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures
from sklearn.decomposition import PCA, FastICA,SparsePCA,KernelPCA
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
train = pd.read_csv(base_input_path+'train.csv')
test = pd.read_csv(base_input_path+'test.csv')

y_train = train['y'].values
id_test = test['ID']

# num_train = len(train)
# df_all = pd.concat([train, test])
# df_all.drop(['ID', 'y'], axis=1, inplace=True)
#
# df_all = pd.get_dummies(df_all, drop_first=True)
#
# train = df_all[:num_train]
# test = df_all[num_train:]


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
# train=train[train.y<250]

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




class StackingCVRegressorAveraged(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds

    def fit(self, X, y):
        self.regr_ = [list() for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        for i, clf in enumerate(self.regressors):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clf)
                self.regr_[i].append(instance)

                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred

        self.meta_regr_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([r.predict(X) for r in regrs]).mean(axis=1)
            for regrs in self.regr_
        ])
        return self.meta_regr_.predict(meta_features)


class StackingCVRegressorRetrained(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        # Create out-of-fold predictions for training meta-model
        for i, regr in enumerate(self.regr_):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(regr)
                instance.fit(X[train_idx], y[train_idx])
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])

        # Train meta-model
        self.meta_regr_.fit(out_of_fold_predictions, y)

        # Retrain base models on all data
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])
        return self.meta_regr_.predict(meta_features)


en = make_pipeline(RobustScaler(), PCA(n_components=125), ElasticNet(alpha=0.001, l1_ratio=0.1))

rf = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25, max_depth=3)

et = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35, max_features=150)

stack_avg = StackingCVRegressorAveraged((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))

stack_retrain = StackingCVRegressorRetrained((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))

results = cross_val_score(en, train.values, y_train, cv=5, scoring='r2')
print("ElasticNet score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(rf, train.values, y_train, cv=5, scoring='r2')
print("RandomForest score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(et, train.values, y_train, cv=5, scoring='r2')
print("ExtraTrees score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(stack_retrain, train.values, y_train, cv=5, scoring='r2')
print("Stacking (retrained) score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(stack_avg, train.values, y_train, cv=5, scoring='r2')
print("Stacking (averaged) score: %.4f (%.4f)" % (results.mean(), results.std()))