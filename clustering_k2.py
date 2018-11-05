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

tit_X_train, tit_y_train, tit_X_test, tit_y_test = load_titanic_data()
ten_X_train, ten_y_train, ten_X_test, ten_y_test = load_tennis_data()

tit_features, tit_labels = load_titanic_data(form="df")
ten_features, ten_labels = load_tennis_data(form="df")

tit_cols = list(tit_features.columns)
ten_cols = list(ten_features.columns)
print(tit_cols)
print(ten_cols)

def accuracy_k2(p, y):
    val1 = float(np.sum(np.absolute(p-y)))/y.shape[0]
    val2 = 1-val1
    return np.amax([val1, val2])

# K=2 TITANIC

pca_titanic = []
ica_titanic = []
rca_titanic = []
rfe_titanic = []

k=2
for dim in range(1, len(tit_cols)+1):
    pca = PCA(n_components=dim)
    ica = FastICA(n_components=dim)
    rca = GaussianRandomProjection(n_components=dim)
    logreg = LogisticRegression()
    rfe = RFE(logreg, n_features_to_select=dim)
    pca_X_train = pca.fit_transform(tit_X_train)
    ica_X_train = ica.fit_transform(tit_X_train)
    rca_X_train = rca.fit_transform(tit_X_train)
    rfe.fit(tit_X_train, tit_y_train)
    rfe_X_train = rfe.transform(tit_X_train)
    em = GaussianMixture(n_components=k)
    em.fit(pca_X_train)
    pca_em_X_train = em.predict(pca_X_train)
    em.fit(ica_X_train)
    ica_em_X_train = em.predict(ica_X_train)
    em.fit(rca_X_train)
    rca_em_X_train = em.predict(rca_X_train)
    em.fit(rfe_X_train)
    rfe_em_X_train = em.predict(rfe_X_train)

    pca_em_X_train = np.array(pca_em_X_train)
    ica_em_X_train = np.array(ica_em_X_train)
    rca_em_X_train = np.array(rca_em_X_train)
    rfe_em_X_train = np.array(rfe_em_X_train)

    pca_titanic.append(accuracy_k2(pca_em_X_train, tit_y_train))
    ica_titanic.append(accuracy_k2(ica_em_X_train, tit_y_train))
    rca_titanic.append(accuracy_k2(rca_em_X_train, tit_y_train))
    rfe_titanic.append(accuracy_k2(rfe_em_X_train, tit_y_train))

plt.plot(range(1, len(tit_cols)+1), pca_titanic, linestyle='-', marker='o', label='PCA')
plt.plot(range(1, len(tit_cols)+1), ica_titanic, linestyle='-', marker='o', label='ICA')
plt.plot(range(1, len(tit_cols)+1), rca_titanic, linestyle='-', marker='o', label='RCA')
plt.plot(range(1, len(tit_cols)+1), rfe_titanic, linestyle='-', marker='o', label='RFE')
plt.title("dim reduc => k=2 => accuracy vs labels - Titanic")
plt.xlabel("number of dimensions")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.savefig("titanic_k2.png")
plt.clf()


pca_tennis = []
ica_tennis = []
rca_tennis = []
rfe_tennis = []

for dim in range(1, len(ten_cols)+1):
    pca = PCA(n_components=dim)
    ica = FastICA(n_components=dim)
    rca = GaussianRandomProjection(n_components=dim)
    logreg = LogisticRegression()
    rfe = RFE(logreg, n_features_to_select=dim)
    pca_X_train = pca.fit_transform(ten_X_train)
    ica_X_train = ica.fit_transform(ten_X_train)
    rca_X_train = rca.fit_transform(ten_X_train)
    rfe.fit(ten_X_train, ten_y_train)
    rfe_X_train = rfe.transform(ten_X_train)
    em = GaussianMixture(n_components=k)
    em.fit(pca_X_train)
    pca_em_X_train = em.predict(pca_X_train)
    em.fit(ica_X_train)
    ica_em_X_train = em.predict(ica_X_train)
    em.fit(rca_X_train)
    rca_em_X_train = em.predict(rca_X_train)
    em.fit(rfe_X_train)
    rfe_em_X_train = em.predict(rfe_X_train)

    pca_em_X_train = np.array(pca_em_X_train)
    ica_em_X_train = np.array(ica_em_X_train)
    rca_em_X_train = np.array(rca_em_X_train)
    rfe_em_X_train = np.array(rfe_em_X_train)

    pca_tennis.append(accuracy_k2(pca_em_X_train, ten_y_train))
    ica_tennis.append(accuracy_k2(ica_em_X_train, ten_y_train))
    rca_tennis.append(accuracy_k2(rca_em_X_train, ten_y_train))
    rfe_tennis.append(accuracy_k2(rfe_em_X_train, ten_y_train))

plt.plot(range(1, len(ten_cols)+1), pca_tennis, linestyle='-', marker='o', label='PCA')
plt.plot(range(1, len(ten_cols)+1), ica_tennis, linestyle='-', marker='o', label='ICA')
plt.plot(range(1, len(ten_cols)+1), rca_tennis, linestyle='-', marker='o', label='RCA')
plt.plot(range(1, len(ten_cols)+1), rfe_tennis, linestyle='-', marker='o', label='RFE')
plt.title("dim reduc => k=2 => accuracy vs labels - Tennis")
plt.xlabel("number of dimensions")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.savefig("tennis_k2.png")
plt.clf()
