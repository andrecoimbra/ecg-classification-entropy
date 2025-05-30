import numpy as np
import sys
import os

from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # Read Data_S.npy and Data_L.npy which contain the features and labels of the whole dataset;
# data_in_S, data_in_L = np.load("Data_S.npy"), np.load(
#     "Data_L.npy"
# )  # data_in_S, data_in_L = np.load('10000_Data_S.npy'), np.load('10000_Data_L.npy')

data_in_S = np.load(f"Data_S_full.npy")
data_in_L = np.load(f"Data_L_full.npy")

entropy_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

entropy_names = [
    "Recurrence Microstates",
    "Approximate Entropy",
    "Sample Entropy",
    "Shannon Entropy",
    "Spectral Entropy",
    "SVD Entropy",
    "Approximate + Sample",
    "Approximate + Shannon",
    "Approximate + Spectral",
    "Approximate + SVD",
    "Sample + Shannon",
    "Sample + Spectral",
    "Sample + SVD",
    "Shannon + Spectral",
    "Shannon + SVD",
    "Spectral + SVD",
]

# Column indices for each entropy or entropy combination
entropy_slices = {
    0: slice(0, 24),  # Recurrence Microstates
    1: slice(24, 36),  # Approximate Entropy
    2: slice(36, 48),  # Sample Entropy
    3: slice(48, 60),  # Shannon Entropy
    4: slice(60, 72),  # Spectral Entropy
    5: slice(72, 84),  # SVD Entropy
    6: np.r_[24:36, 36:48],  # Approximate + Sample
    7: np.r_[24:36, 48:60],  # Approximate + Shannon
    8: np.r_[24:36, 60:72],  # Approximate + Spectral
    9: np.r_[24:36, 72:84],  # Approximate + SVD
    10: np.r_[36:48, 48:60],  # Sample + Shannon
    11: np.r_[36:48, 60:72],  # Sample + Spectral
    12: np.r_[36:48, 72:84],  # Sample + SVD
    13: np.r_[48:60, 60:72],  # Shannon + Spectral
    14: np.r_[48:60, 72:84],  # Shannon + SVD
    15: np.r_[60:72, 72:84],  # Spectral + SVD
}

selected_columns = entropy_slices[entropy_id]
data_in_S = data_in_S[:, selected_columns]

# Select the groups of interest
X = []
Y = []
group_list = np.array([0, 1, 2, 3])  # ex: SR, SB, AFIB, ST

for i in range(len(data_in_L)):
    if data_in_L[i] in group_list:
        Y.append(data_in_L[i])
        X.append(data_in_S[i])

X = np.array(X)
Y = np.array(Y)

# shuffling the dataset
seed = 1001
np.random.seed(seed)
np.random.shuffle(X)

np.random.seed(seed)
np.random.shuffle(Y)

# Choose the classifier
classifier = int(sys.argv[2]) if len(sys.argv) > 2 else 4

classifier_list = ["knn", "dtree", "rfm", "svm", "ANN"]

print(
    f"Multiclass Classification using {entropy_names[entropy_id]} and {classifier_list[classifier]}."
)


def build_ann(input_dim, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# K-fold step
K = 10
num_val_samples = len(X) // K
all_scores = []


for k in range(K):
    print("processing fold #", k)
    test_data = X[k * num_val_samples : (k + 1) * num_val_samples]
    test_label = Y[k * num_val_samples : (k + 1) * num_val_samples]

    train_data = np.concatenate(
        [X[: k * num_val_samples], X[(k + 1) * num_val_samples :]], axis=0
    )
    train_label = np.concatenate(
        [Y[: k * num_val_samples], Y[(k + 1) * num_val_samples :]], axis=0
    )

    if classifier == 0:  # knn
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)  # ,test_label)
        acc = accuracy_score(prediction, test_label)

    if classifier == 1:  # dtree
        model = DecisionTreeClassifier()
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)  # ,test_label)
        acc = accuracy_score(prediction, test_label)

    if classifier == 2:  # rfm
        model = RandomForestClassifier()
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)  # ,test_label)
        acc = accuracy_score(prediction, test_label)

    if classifier == 3:  # SVM
        model = svm.SVC(kernel="rbf")
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)  # ,test_label)
        acc = accuracy_score(prediction, test_label)

    if classifier == 4:  # ANN
        num_classes = len(np.unique(Y))
        train_label = to_categorical(train_label, num_classes=num_classes)
        test_label = to_categorical(test_label, num_classes=num_classes)
        model = build_ann(input_dim=X.shape[1], num_classes=num_classes)
        model.fit(train_data, train_label, epochs=1000, batch_size=256, verbose=0)
        _, acc = model.evaluate(test_data, test_label, verbose=0)

    print(acc)
    all_scores.append(acc)

print(
    f"### {entropy_names[entropy_id]} | Classifier: {classifier_list[classifier]} ###",
    "ACC= %.2f +- %.2f" % (np.mean(all_scores) * 100, np.std(all_scores) * 100),
    sep="\n",
)
