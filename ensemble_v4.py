import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA,SparsePCA,KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score

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

base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
train = pd.read_csv(base_input_path+'train.csv')
test = pd.read_csv(base_input_path+'test.csv')


for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

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

# usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
# finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays)
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

'''Train the xgb model then predict the test data'''

xgb_params = {
    'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,  # base prediction = mean(target)
    'silent': 1
}

# prepare dict of params for xgboost to run with


# NOTE: Make sure that the class is labeled 'class' in the data file

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

num_boost_rounds = 1350
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain,
                  num_boost_round=num_boost_rounds,
                  # early_stopping_rounds=50,
                  verbose_eval=10)
y_pred = model.predict(dtest)

'''Train the stacked models then predict the test data'''

# Stacking with average
en = make_pipeline(RobustScaler(), PCA(n_components=12), ElasticNet(alpha=0.001, l1_ratio=0.1))

rf = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25, max_depth=3)

et = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35, max_features=150)

# stack_avg = StackingCVRegressorAveraged((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))

# stack_retrain = StackingCVRegressorRetrained((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25, max_depth=3)),
    StackingEstimator(estimator=ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35, max_features=150)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(
        estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55,
                                            min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()
)


stacked_pipeline_avg = StackingCVRegressorAveraged((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))

stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

stacked_pipeline_avg.fit(finaltrainset, y_train)
results_stacked_avg = stacked_pipeline_avg.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
# y_pred_all_model = stacked_pipeline.predict(finaltrainset) * 0.2855 + model.predict(dtrain) * 0.7145
y_pred_all_stacking = stacked_pipeline.predict(finaltrainset) * 0.2855 +results_stacked_avg * 0.2855+ model.predict(dtrain) * 0.429

model_r2_score = r2_score(y_train, y_pred_all_stacking)
print(model_r2_score)

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
# sub['y'] = y_pred * 0.75 + results * 0.25
sub['y'] = y_pred * 0.50 + results * 0.25 + results_stacked_avg * 0.25
sub.to_csv(base_output_path+'ensemble_v4_r2_score_{}.csv'.format(model_r2_score), index=False)
