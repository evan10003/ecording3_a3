from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from data import load_tennis_data, load_titanic_data
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np

# IMPORT DATA ETC

tit_X_train, tit_y_train, tit_X_test, tit_y_test = load_titanic_data()
tit_df = load_titanic_data(form="original df")
tit_features, tit_labels = load_titanic_data(form="df")

tit_cols = list(tit_features.columns)
print(tit_cols)
tit_l = len(tit_X_train)

# K MEANS
ks = [2,4,6,8,10,12,14,16,18,20,22,24,26]
ks = list(range(2,27))
wss = []
wss_em = []
log_like_em = []
for k in ks:
    if True:
        model = KMeans(n_clusters=k)
        model.fit(tit_X_train)
        wss.append(model.inertia_)

# EM
for k in ks:
    if True:
        em = GaussianMixture(n_components=k)
        em.fit(tit_X_train)
        labels = np.array(em.predict(tit_X_train))
        centers = np.array(em.means_)
        sum = 0
        for idx in range(len(labels)):
            label = labels[idx]
            center = centers[label]
            point = tit_X_train[idx]
            sum += np.sum((point-center)**2)
        wss_em.append(sum)
        log_like_em.append(em.lower_bound_)

# ELBOW  CURVES

if True:
    plt.scatter(ks, wss)
    plt.title("k means elbow curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("sum of squares (wss)")
    plt.savefig("kmeans_elbow_titanic.png")
    plt.clf()

if True:
    plt.scatter(ks, wss_em)
    plt.title("EM elbow curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("sum of squares (wss)")
    plt.savefig("em_elbow_titanic.png")
    plt.clf()

if True:
    plt.scatter(ks, log_like_em)
    plt.title("EM log likelihood curve for Titanic")
    plt.xlabel("number of clusters")
    plt.ylabel("log likelihood")
    plt.savefig("em_log_like_titanic.png")
    plt.clf()


# FIND CLASSIFICATION ACCURACY FOR K=2

k=2

if True:
    km = KMeans(n_clusters=k)
    km.fit(tit_X_train)
    km_labels1 = np.array(km.predict(tit_X_train))
    #km_labels2 = np.array([np.absolute(1-x) for x in km_labels1])
    em = GaussianMixture(n_components=k)
    em.fit(tit_X_train)
    em_labels1 = np.array(em.predict(tit_X_train))
    #em_labels2 = np.array([np.absolute(1-x) for x in em_labels1])

    km1_true_pos = 0
    km1_true_neg = 0
    km1_false_neg = 0
    km1_false_pos = 0
    km2_true_pos = 0
    km2_true_neg = 0
    km2_false_neg = 0
    km2_false_pos = 0
    em1_true_pos = 0
    em1_true_neg = 0
    em1_false_neg = 0
    em1_false_pos = 0
    em2_true_pos = 0
    em2_true_neg = 0
    em2_false_neg = 0
    em2_false_pos = 0

    em_true_pos = 0
    em_true_neg = 0
    em_false_pos = 0
    em_false_neg = 0
    km_true_pos = 0
    km_true_neg = 0
    km_false_pos = 0
    km_false_neg = 0

    for i in range(len(tit_y_train)):
        if km_labels1[i] == 0:
            if tit_y_train[i] == 0:
                km1_true_neg += 1
                km2_false_pos += 1
            else:
                km1_false_neg += 1
                km2_true_pos += 1
        else:
            if tit_y_train[i] == 0:
                km1_false_pos += 1
                km2_true_neg += 1
            else:
                km1_true_pos += 1
                km2_false_neg += 1
        if em_labels1[i] == 0:
            if tit_y_train[i] == 0:
                em1_true_neg += 1
                em2_false_pos += 1
            else:
                em1_false_neg += 1
                em2_true_pos += 1
        else:
            if tit_y_train[i] == 0:
                em1_false_pos += 1
                em2_true_neg += 1
            else:
                em1_true_pos += 1
                em2_false_neg += 1

    if em1_true_pos + em1_true_neg > em2_true_pos + em2_true_neg:
        em_true_pos = float(em1_true_pos)
        em_true_neg = float(em1_true_neg)
        em_false_pos = float(em1_false_pos)
        em_false_neg = float(em1_false_neg)
    else:
        em_true_pos = float(em2_true_pos)
        em_true_neg = float(em2_true_neg)
        em_false_pos = float(em2_false_pos)
        em_false_neg = float(em2_false_neg)
    if km1_true_pos + km1_true_neg > km2_true_pos + km2_true_neg:
        km_true_pos = float(km1_true_pos)
        km_true_neg = float(km1_true_neg)
        km_false_pos = float(km1_false_pos)
        km_false_neg = float(km1_false_neg)
    else:
        km_true_pos = float(km2_true_pos)
        km_true_neg = float(km2_true_neg)
        km_false_pos = float(km2_false_pos)
        km_false_neg = float(km2_false_neg)

    plt.xticks([0.25, 0.75], ['KMeans', 'EM'])
    plt.scatter([0.25, 0.75], [(km_true_pos+km_true_neg)/len(tit_y_train), (em_true_pos+em_true_neg)/len(tit_y_train)], label='accuracy')
    plt.scatter([0.25, 0.75], [km_true_pos/(km_true_pos+km_false_pos), em_true_pos/(em_true_pos+em_false_pos)], label='true positive rate')
    plt.scatter([0.25, 0.75], [km_true_neg/(km_true_neg+km_false_neg), em_true_neg/(em_true_neg+em_false_neg)], label='true negative rate')

    plt.title("k=2 KMeans and EM accuracy for Titanic Dataset")
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig("k2_accuracy_titanic.png")
    plt.clf()



