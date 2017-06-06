import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv("./data/input/train.csv").set_index("ID")

# identify constant features by looking at the standard deviation (check id std ==0.0)
desc = df.describe().transpose()
columns_to_drop = desc.loc[desc["std"] == 0].index.values
df.drop(columns_to_drop, axis=1, inplace=True)

# check which column has been dropped
print(columns_to_drop)

# do one hot encoding for categorical columns
df08 = df[["X{}".format(x) for x in range(9) if x != 7]]

tot_cardinality = 0
for c in df08.columns.values:
    cardinality = len(df08[c].unique())
    print(c, cardinality)
    tot_cardinality += cardinality
print(tot_cardinality)

df = pd.get_dummies(df, columns=["X{}".format(x) for x in range(9) if x != 7])

# drop outliers in target varaible
df.drop(df.loc[df["y"] > 250].index, inplace=True)

#tsne analysis
tsne2 = TSNE(n_components=2)
tsne2_results = tsne2.fit_transform(df.drop(["y"], axis=1))

f, ax = plt.subplots(figsize=(20,15))
points = ax.scatter(tsne2_results[:,0], tsne2_results[:,1], c=df.y, s=50, cmap=cmap)
f.colorbar(points)
plt.show()

