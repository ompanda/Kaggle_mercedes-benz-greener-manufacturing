import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, SparsePCA, KernelPCA
from sklearn.decomposition import TruncatedSVD
color = sns.color_palette()

# Load the data
base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
train_df = pd.read_csv(base_input_path+'train.csv')
test_df = pd.read_csv(base_input_path+'test.csv')

print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)


# Do label encoding
for c in train_df.columns:
    if train_df[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train_df[c].values) + list(test_df[c].values))
        train_df[c] = lbl.transform(list(train_df[c].values))
        test_df[c] = lbl.transform(list(test_df[c].values))



n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train_df.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test_df)

# # PCA
# pca = PCA(n_components=n_comp, random_state=420)
# pca2_results_train = pca.fit_transform(train_df.drop(["y"], axis=1))
# pca2_results_test = pca.transform(test_df)

#sparse PCA
spca = SparsePCA(n_components=n_comp, random_state=420)
spca2_results_train = spca.fit_transform(train_df.drop(["y"], axis=1))
spca2_results_test = spca.transform(test_df)

#Kernel PCA
kpca = KernelPCA(n_components=n_comp, random_state=420)
kpca2_results_train = kpca.fit_transform(train_df.drop(["y"], axis=1))
kpca2_results_test = kpca.transform(test_df)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train_df.drop(["y"], axis=1))
ica2_results_test = ica.transform(test_df)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train_df.drop(["y"], axis=1))
grp_results_test = grp.transform(test_df)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train_df.drop(["y"], axis=1))
srp_results_test = srp.transform(test_df)

usable_columns = list(set(train_df.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    # train_df['pca_' + str(i)] = pca2_results_train[:, i - 1]
    # test_df['pca_' + str(i)] = pca2_results_test[:, i - 1]


    train_df['spca_' + str(i)] = spca2_results_train[:, i - 1]
    test_df['spca_' + str(i)] = spca2_results_test[:, i - 1]

    train_df['kpca_' + str(i)] = kpca2_results_train[:, i - 1]
    test_df['kpca_' + str(i)] = kpca2_results_test[:, i - 1]

    train_df['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test_df['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train_df['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test_df['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train_df['grp_' + str(i)] = grp_results_train[:, i - 1]
    test_df['grp_' + str(i)] = grp_results_test[:, i - 1]

    train_df['srp_' + str(i)] = srp_results_train[:, i - 1]
    test_df['srp_' + str(i)] = srp_results_test[:, i - 1]

y_train = train_df['y'].values
y_mean = np.mean(y_train)
id_test = test_df['ID'].values
finaltrainset = train_df[usable_columns].values
finaltestset = test_df[usable_columns].values

from tpot import TPOTRegressor
auto_classifier = TPOTRegressor(generations=3, population_size=8, verbosity=2)
from sklearn.model_selection import train_test_split

# Split training data to train and validate
X_train, X_valid, y_train, y_valid = train_test_split(finaltrainset, y_train,
                                                    train_size=0.75, test_size=0.25)

auto_classifier.fit(X_train, y_train)

cv_score= auto_classifier.score(X_valid, y_valid)

print("The cross-validation accuracy")
print(cv_score)




# we need access to the pipeline to get the probabilities
test_result = auto_classifier.predict(finaltestset)
sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = test_result

sub.to_csv(base_output_path+'tpot_analysis_{}.csv'.format(cv_score), index=False)

auto_classifier.export(base_output_path+'tpot_pipeline.py')

