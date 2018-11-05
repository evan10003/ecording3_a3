from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from data import load_tennis_data, load_titanic_data
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import kurtosis

NEGINF = -float("inf")

# IMPORT DATA

ten_X_train, ten_y_train, ten_X_test, ten_y_test = load_tennis_data()
tit_X_train, tit_y_train, tit_X_test, tit_y_test = load_titanic_data()

ten_features, ten_labels = load_tennis_data(form="df")
tit_features, tit_labels = load_titanic_data(form="df")

tit_cols = list(tit_features.columns)
ten_cols = list(ten_features.columns)

# NO DIM REDUC CLUSTERING

tit_sil_em = []
ten_sil_em = []

ks = range(2, 20)
for k in ks:
    em = GaussianMixture(n_components=k)
    em.fit(tit_X_train)
    tit_sil_em.append(silhouette_score(tit_X_train, em.predict(tit_X_train)))

for k in ks:
    em = GaussianMixture(n_components=k)
    em.fit(ten_X_train)
    ten_sil_em.append(silhouette_score(ten_X_train, em.predict(ten_X_train)))

np.savetxt('no_dem_reduc_sil_em_tennis', ten_sil_em)
np.savetxt('no_dem_reduc_sil_em_titanic', tit_sil_em)










