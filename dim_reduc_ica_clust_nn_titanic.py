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


# ICA

wss_km = []
log_like_em = []
sil_km = []
sil_em = []
ica_models = []
km_models = []
em_models = []

ks = range(2, 20)
for dim in range(1, len(tit_cols)):
    wss_km.append([])
    log_like_em.append([])
    sil_km.append([])
    sil_em.append([])
    ica = FastICA(n_components=dim)
    ica.fit(X_train)
    ica_models.append(ica)
    ica_X_train = ica.transform(X_train)
    km_models.append([])
    em_models.append([])
    for k in ks:
        km = KMeans(n_clusters=k)
        em = GaussianMixture(n_components=k)
        em.fit(ica_X_train)
        km.fit(ica_X_train)
        km_models[-1].append(km)
        em_models[-1].append(em)
        sil_em[-1].append(silhouette_score(ica_X_train, em.predict(ica_X_train)))
        sil_km[-1].append(silhouette_score(ica_X_train, np.array(km.labels_)))
        log_like_em[-1].append(em.lower_bound_)
        wss_km[-1].append(km.inertia_)

np.savetxt('ica_wss_km_titanic', wss_km)
np.savetxt('ica_log_like_em_titanic', log_like_em)
np.savetxt('ica_sil_km_titanic', sil_km)
np.savetxt('ica_sil_em_titanic', sil_em)

# if False:
#     for j in range(len(tit_cols)-1):
#     #for j in range(1):
#         plt.plot(ks, wss_km[j], linestyle='-', marker='o', label=str(j+1))
#     plt.title("ICA kmeans ss curves - Titanic")
#     plt.xlabel("number of clusters")
#     plt.ylabel("sum of squares")
#     plt.legend(loc="best")
#     plt.savefig("ica_km_wss_titanic.png")
#     plt.clf()
#     for j in range(len(tit_cols)-1):
#     #for j in range(1):
#         plt.plot(ks, log_like_em[j], linestyle='-', marker='o', label=str(j+1))
#     plt.title("ICA EM log likelihood curves - Titanic")
#     plt.xlabel("number of clusters")
#     plt.ylabel("log likelihood")
#     plt.savefig("ica_em_log_like_titanic.png")
#     plt.legend(loc="best")
#     plt.clf()
#     for j in range(len(tit_cols)-1):
#     #for j in range(3):
#         plt.plot(ks, sil_em[j], linestyle='-', marker='o', label=str(j+1))
#     plt.title("ICA EM silhouette curves - Titanic")
#     plt.xlabel("number of clusters")
#     plt.ylabel("silhouette score")
#     plt.savefig("ica_em_sil_titanic.png")
#     plt.legend(loc="best")
#     plt.clf()
#     for j in range(len(tit_cols)-1):
#     #for j in range(3):
#         plt.plot(ks, sil_km[j], linestyle='-', marker='o', label=str(j+1))
#     plt.title("ICA kmeans silhouette curvess - Titanic")
#     plt.xlabel("number of clusters")
#     plt.ylabel("silhouette score")
#     plt.savefig("ica_km_sil_titanic.png")
#     plt.legend(loc="best")
#     plt.clf()


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


plain_best_nn, plain_best_nn_score, plain_best_nn_h = best_nn(X_train, y_train, X_test, y_test)

ica_km_nn_scores = []
ica_em_nn_scores = []
ica_nn_scores = []
ks = range(2, 20)
for j in range(len(tit_cols)-1):
    ica = ica_models[j]
    ica_X_train = ica.transform(X_train)
    ica_X_test = ica.transform(X_test)
    ica_km_nn_scores.append([])
    ica_em_nn_scores.append([])
    _, new_score, _ = best_nn(ica_X_train, y_train, ica_X_test, y_test)
    ica_nn_scores.append(new_score)
    for k in range(18):
        km = km_models[j][k]
        em = em_models[j][k]
        ica_km_X_train = np.array(labels_to_matrix(list(km.labels_), k+2))
        ica_em_X_train = np.array(labels_to_matrix(list(em.predict(ica_X_train)), k+2))
        ica_km_X_test = np.array(labels_to_matrix(list(km.predict(ica_X_test)), k+2))
        ica_em_X_test = np.array(labels_to_matrix(list(em.predict(ica_X_test)), k+2))
        _, new_score, _ = best_nn(ica_km_X_train, y_train, ica_km_X_test, y_test)
        ica_km_nn_scores[-1].append(new_score)
        _, new_score, _ = best_nn(ica_em_X_train, y_train, ica_em_X_test, y_test)
        ica_em_nn_scores[-1].append(new_score)

np.savetxt('ica_nn_scores_titanic', ica_nn_scores)
np.savetxt('ica_em_nn_scores_titanic', ica_em_nn_scores)
np.savetxt('ica_km_nn_scores_titanic', ica_km_nn_scores)

plt.plot(range(1, len(tit_cols)), ica_nn_scores, linestyle='-', marker='o')
plt.title("ICA nn scores - Titanic")
plt.xlabel("number of components")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("ica_nn_titanic.png")
plt.clf()

for j in range(len(tit_cols)-1):
    plt.plot(ks, ica_km_nn_scores[j], linestyle='-', marker='o', label=str(j+1))
plt.title("ICA kmeans nn scores - Titanic")
plt.xlabel("number of clusters")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("ica_km_nn_titanic.png")
plt.clf()

for j in range(len(tit_cols)-1):
    plt.plot(ks, ica_em_nn_scores[j], linestyle='-', marker='o', label=str(j+1))
plt.title("ICA EM nn scores - Titanic")
plt.xlabel("number of clusters")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("ica_em_nn_titanic.png")
plt.clf()



















