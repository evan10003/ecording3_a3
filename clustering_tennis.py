from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from data import load_tennis_data, load_titanic_data
import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np

# IMPORT DATA ETC

ten_X_train1, ten_y_train1, ten_X_test, ten_y_test = load_tennis_data()
ten_df = load_tennis_data(form="original df")
ten_features, ten_labels = load_tennis_data(form="df")

val_cutoff2 = len(ten_X_train1)-1000
val_cutoff1 = val_cutoff2-1000

X_val1 = ten_X_train1[val_cutoff1:val_cutoff2]
y_val1 = ten_y_train1[val_cutoff1:val_cutoff2]
X_val2 = ten_X_train1[val_cutoff2:]
y_val2 = ten_y_train1[val_cutoff2:]

# ten_X_train = ten_X_train1[:val_cutoff1]
# ten_y_train = ten_y_train1[:val_cutoff1]
ten_X_train = ten_X_train1[:2000]
ten_y_train = ten_y_train1[:2000]


ten_cols = list(ten_features.columns)
print(ten_cols)
ten_l = len(ten_X_train)

# K MEANS

ks = [2,4,6,8,10,12,14,16,18,20,22,24]
ks = list(range(2, 24))
wss = []
wss_em = []
log_like_em = []
for k in ks:
    if False:
        model = KMeans(n_clusters=k)
        model.fit(ten_X_train)
        wss.append(model.inertia_)

# EM

for i in range(5):
    wss_em.append([])
    log_like_em.append([])
    for k in ks:
        if False:
            em = GaussianMixture(n_components=k)
            em.fit(ten_X_train)
            labels = np.array(em.predict(ten_X_train))
            centers = np.array(em.means_)
            sum = 0
            for idx in range(len(labels)):
                label = labels[idx]
                center = centers[label]
                point = ten_X_train[idx]
                sum += np.sum((point-center)**2)
            wss_em[-1].append(sum)
            log_like_em[-1].append(em.lower_bound_)

# ELBOW CURVES

if False:
    plt.scatter(ks, wss)
    plt.title("k means elbow curve for Tennis")
    plt.xlabel("number of clusters")
    plt.ylabel("sum of squares")
    plt.savefig("kmeans_elbow_tennis.png")
    plt.clf()

if False:
    for i in range(5):
        plt.scatter(ks, wss_em[i])
    plt.title("EM elbow curve for Tennis")
    plt.xlabel("number of clusters")
    plt.ylabel("sum of squares")
    plt.savefig("em_elbow_tennis.png")
    plt.clf()

if False:
    for i in range(5):
        plt.scatter(ks, log_like_em[i])
    plt.title("EM log likelihood curve for Tennis")
    plt.xlabel("number of clusters")
    plt.ylabel("log likelihood")
    plt.savefig("em_log_like_tennis.png")
    plt.clf()

# FIND CLASSIFICATION ACCURACY FOR K=2

k=2

if True:
    km = KMeans(n_clusters=k)
    km.fit(ten_X_train)
    km_labels1 = np.array(km.predict(ten_X_train))
    #km_labels2 = np.array([np.absolute(1-x) for x in km_labels1])
    em = GaussianMixture(n_components=k)
    em.fit(ten_X_train)
    em_labels1 = np.array(em.predict(ten_X_train))
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

    for i in range(len(ten_y_train)):
        if km_labels1[i] == 0:
            if ten_y_train[i] == 0:
                km1_true_neg += 1
                km2_false_pos += 1
            else:
                km1_false_neg += 1
                km2_true_pos += 1
        else:
            if ten_y_train[i] == 0:
                km1_false_pos += 1
                km2_true_neg += 1
            else:
                km1_true_pos += 1
                km2_false_neg += 1
        if em_labels1[i] == 0:
            if ten_y_train[i] == 0:
                em1_true_neg += 1
                em2_false_pos += 1
            else:
                em1_false_neg += 1
                em2_true_pos += 1
        else:
            if ten_y_train[i] == 0:
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
    plt.scatter([0.25, 0.75], [(km_true_pos+km_true_neg)/len(ten_y_train), (em_true_pos+em_true_neg)/len(ten_y_train)], label='accuracy')
    plt.scatter([0.25, 0.75], [km_true_pos/(km_true_pos+km_false_pos), em_true_pos/(em_true_pos+em_false_pos)], label='true positive rate')
    plt.scatter([0.25, 0.75], [km_true_neg/(km_true_neg+km_false_neg), em_true_neg/(em_true_neg+em_false_neg)], label='true negative rate')

    plt.title("k=2 KMeans and EM accuracy for Tennis Dataset")
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig("k2_accuracy_tennis.png")
    plt.clf()






















