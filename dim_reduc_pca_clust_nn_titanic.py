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

X_train, y_train, X_test, y_test = load_titanic_data()
tit_X_train, tit_y_train, tit_X_test, tit_y_test = load_titanic_data()

tit_df = load_titanic_data(form="original df")
tit_features, tit_labels = load_titanic_data(form="df")

tit_cols = list(tit_features.columns)
print(tit_cols)


# PCA

wss_km = []
log_like_em = []
sil_km = []
sil_em = []
pca_models = []
km_models = []
em_models = []

ks = range(2, 20)
for dim in range(1, len(tit_cols)):
    wss_km.append([])
    log_like_em.append([])
    sil_km.append([])
    sil_em.append([])
    pca = PCA(n_components=dim)
    pca.fit(X_train)
    pca_models.append(pca)
    pca_X_train = pca.transform(X_train)
    km_models.append([])
    em_models.append([])
    for k in ks:
        km = KMeans(n_clusters=k)
        em = GaussianMixture(n_components=k)
        em.fit(pca_X_train)
        km.fit(pca_X_train)
        km_models[-1].append(km)
        em_models[-1].append(em)
        sil_em[-1].append(silhouette_score(pca_X_train, em.predict(pca_X_train)))
        sil_km[-1].append(silhouette_score(pca_X_train, np.array(km.labels_)))
        log_like_em[-1].append(em.lower_bound_)
        wss_km[-1].append(km.inertia_)

for j in range(len(tit_cols)-1):
#for j in range(1):
    plt.plot(ks, wss_km[j], linestyle='-', marker='o', label=str(j+1))
plt.title("PCA kmeans ss curves - Titanic")
plt.xlabel("number of clusters")
plt.ylabel("sum of squares")
plt.legend(loc="best")
plt.savefig("pca_km_wss_titanic.png")
plt.clf()
for j in range(len(tit_cols)-1):
#for j in range(1):
    plt.plot(ks, log_like_em[j], linestyle='-', marker='o', label=str(j+1))
plt.title("PCA EM log likelihood curves - Titanic")
plt.xlabel("number of clusters")
plt.ylabel("log likelihood")
plt.savefig("pca_em_log_like_titanic.png")
plt.legend(loc="best")
plt.clf()
for j in range(len(tit_cols)-1):
#for j in range(3):
    plt.plot(ks, sil_em[j], linestyle='-', marker='o', label=str(j+1))
plt.title("PCA EM silhouette curves - Titanic")
plt.xlabel("number of clusters")
plt.ylabel("silhouette score")
plt.savefig("pca_em_sil_titanic.png")
plt.legend(loc="best")
plt.clf()
for j in range(len(tit_cols)-1):
#for j in range(3):
    plt.plot(ks, sil_km[j], linestyle='-', marker='o', label=str(j+1))
plt.title("PCA kmeans silhouette curvess - Titanic")
plt.xlabel("number of clusters")
plt.ylabel("silhouette score")
plt.savefig("pca_km_sil_titanic.png")
plt.legend(loc="best")
plt.clf()


np.savetxt('pca_wss_km_titanic', wss_km)
np.savetxt('pca_log_like_em_titanic', log_like_em)
np.savetxt('pca_sil_km_titanic', sil_km)
np.savetxt('pca_sil_em_titanic', sil_em)


# NN

# Function to lightly tune NN

def best_nn(X, y, X2, y2):
    max_score = NEGINF
    best_model = None
    best_h = None
    for h in [1,2,3,5,7,9,11,13,15]:
        nn = MLPClassifier(batch_size=16, hidden_layer_sizes=h, alpha=0.001, learning_rate_init=0.01)
        score = np.mean(np.array(cross_val_score(nn, X, y, cv=10)))
        if score > max_score:
            max_score = score
            best_model = nn
            best_h = h
    best_model.fit(X, y)
    new_score = best_model.score(X2, y2)
    return best_model, new_score, best_h

def labels_to_matrix(labels, k):
    matrix = np.zeros((len(labels),k-1))
    for i in range(len(labels)):
        label = labels[i]
        if label != 0:
            matrix[i,label-1] = 1
    return matrix

if False:
    # Plain NN

    plain_best_nn, plain_best_nn_score, plain_best_nn_h = best_nn(X_train, y_train, X_test, y_test)

    pca_km_nn_scores = []
    pca_em_nn_scores = []
    pca_nn_scores = []
    ks = range(2, 20)
    for j in range(len(tit_cols)-1):
        pca = pca_models[j]
        pca_X_train = pca.transform(X_train)
        pca_X_test = pca.transform(X_test)
        pca_km_nn_scores.append([])
        pca_em_nn_scores.append([])
        _, new_score, _ = best_nn(pca_X_train, y_train, pca_X_test, y_test)
        pca_nn_scores.append(new_score)
        for k in range(18):
            km = km_models[j][k]
            em = em_models[j][k]
            pca_km_X_train = np.array(labels_to_matrix(list(km.labels_), k+2))
            pca_em_X_train = np.array(labels_to_matrix(list(em.predict(pca_X_train)), k+2))
            pca_km_X_test = np.array(labels_to_matrix(list(km.predict(pca_X_test)), k+2))
            pca_em_X_test = np.array(labels_to_matrix(list(em.predict(pca_X_test)), k+2))
            _, new_score, _ = best_nn(pca_km_X_train, y_train, pca_km_X_test, y_test)
            pca_km_nn_scores[-1].append(new_score)
            _, new_score, _ = best_nn(pca_em_X_train, y_train, pca_em_X_test, y_test)
            pca_em_nn_scores[-1].append(new_score)

    np.savetxt('pca_nn_scores_titanic', pca_nn_scores)
    np.savetxt('pca_em_nn_scores_titanic', pca_em_nn_scores)
    np.savetxt('pca_km_nn_scores_titanic', pca_km_nn_scores)

    plt.plot(range(1, len(tit_cols)), pca_nn_scores, linestyle='-', marker='o')
    plt.title("PCA nn scores - Titanic")
    plt.xlabel("number of components")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("pca_nn_titanic.png")
    plt.clf()

    for j in range(len(tit_cols)-1):
        plt.plot(ks, pca_km_nn_scores[j], linestyle='-', marker='o', label=str(j+1))
    plt.title("PCA kmeans nn scores - Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("pca_km_nn_titanic.png")
    plt.clf()

    for j in range(len(tit_cols)-1):
        plt.plot(ks, pca_em_nn_scores[j], linestyle='-', marker='o', label=str(j+1))
    plt.title("PCA EM nn scores - Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("pca_em_nn_titanic.png")
    plt.clf()





























# POST PCA CLUSTERING

# VARY DIMS and KS SILHOUETTE

if False:
    ks = list(range(2, 24))
    for dim in range(1, 9):
        sil_many_dims = []
        pca = PCA(n_components=dim)
        pca.fit(X_train)
        pca_X_train = pca.transform(X_train)

        for k in ks:
            km = KMeans(n_clusters=k)
            km.fit(pca_X_train)
            labels = km.predict(pca_X_train)
            sil_many_dims.append(silhouette_score(pca_X_train, labels))
        plt.scatter(ks, sil_many_dims, label='dim '+ str(dim))
    plt.title("PCA --> kmeans --> silhouette scores - Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("silhouette scores")
    plt.legend()
    plt.savefig("pca_many_dims_kmeans_sil_titanic.png")
    plt.clf()


# PCA DIM 3

pca = PCA(n_components=3)
pca.fit(X_train)
pca_X_train = pca.transform(X_train)

# KMEANS
ks = list(range(2, 24))
wss = []
wss_em = []
log_like_em = []
sil_km = []
sil_em = []
for k in ks:
    if False:
        km = KMeans(n_clusters=k)
        km.fit(pca_X_train)
        wss.append(km.inertia_)
        labels = km.predict(pca_X_train)
        sil_km.append(silhouette_score(pca_X_train, labels))

# EM

#for i in range(5):
for k in ks:
    if False:
        em = GaussianMixture(n_components=k)
        em.fit(pca_X_train)
        labels = np.array(em.predict(pca_X_train))
        centers = np.array(em.means_)
        sum = 0
        for idx in range(len(labels)):
            label = labels[idx]
            center = centers[label]
            point = pca_X_train[idx]
            sum += np.sum((point-center)**2)
        wss_em.append(sum)
        log_like_em.append(em.lower_bound_)
        sil_em.append(silhouette_score(pca_X_train, labels))


# POST PCA ELBOW CURVES ETC

if False:
    plt.scatter(ks, wss)
    plt.title("PCA dim=3 k means elbow curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("sum of squares")
    plt.savefig("pca_d3_kmeans_elbow_titanic.png")
    plt.clf()

if False:
#    for i in range(5):
    plt.scatter(ks, wss_em)
    plt.title("PCA dim=3 EM elbow curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("sum of squares")
    plt.savefig("pca_d3_em_elbow_titanic.png")
    plt.clf()

if False:
    plt.scatter(ks, sil_km)
    plt.title("PCA dim=3 k means silhouette curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("silhouette score")
    plt.savefig("pca_d3_kmeans_sil_titanic.png")
    plt.clf()

if False:
#    for i in range(5):
    plt.scatter(ks, sil_em)
    plt.title("PCA dim=3 EM silhouette curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("silhouette score")
    plt.savefig("pca_d3_em_sil_titanic.png")
    plt.clf()

if False:
#    for i in range(5):
    plt.scatter(ks, log_like_em)
    plt.title("PCA dim=3 EM log likelihood curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("log likelihood")
    plt.savefig("pca_d3_em_log_like_titanic.png")
    plt.clf()









