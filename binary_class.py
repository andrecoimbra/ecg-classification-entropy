import numpy as np
import sys

from keras import models, layers

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score


# Read Data_S.npy and Data_L.npy which contain the features and labels of the whole dataset;
# data_in_S,data_in_L = np.load('Data_S.npy'),np.load('Data_L.npy')
# data_in_S, data_in_L = np.load("10000_Data_S.npy"), np.load("10000_Data_L.npy")

data_in_S = np.load(f"Data_S_full.npy")
data_in_L = np.load(f"Data_L_full.npy")

# Check for required arguments
if len(sys.argv) < 2:
    print("Please, you need to select a group to be compared with the SR rhythms")
    print("Usage: python binary_class.py <group> <entropy> <classifier>")
    sys.exit(1)

# Select group to compare against SR
G = int(sys.argv[1])
group_list = np.array([0, G])

# List of diagnostic labels
diag_list = np.array(
    ["SR", "SB", "AFIB", "ST", "SVT", "AF", "SI", "AT", "AVNRT", "AVRT", "SAAWR"]
)

# Entropy measure selection
entropy_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Names of entropy measures and their combinations
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

# Select only the columns corresponding to the chosen entropy measure(s)
selected_columns = entropy_slices[entropy_id]
data_in_S = data_in_S[:, selected_columns]

# Select only the samples of interest (SR and target group)
X = []
Y = []

for i in range(len(data_in_L)):
    if data_in_L[i] in group_list:
        X.append(data_in_S[i])
        Y.append(1 if data_in_L[i] > 0 else 0)

X = np.array(X)
Y = np.array(Y)

# Shuffle the dataset
seed = 1001
np.random.seed(seed)
np.random.shuffle(X)

np.random.seed(seed)
np.random.shuffle(Y)

# Select the classifier
classifier = int(sys.argv[3]) if len(sys.argv) > 3 else 4
classifier_list = ["knn", "dtree", "rfm", "svm", "ANN"]

print(
    f"Comparing SR with {diag_list[G]} using {entropy_names[entropy_id]} and {classifier_list[classifier]} as classifier."
)


# Build a simple feedforward neural network
def build_ann(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# K-fold cross-validation
K = 10
num_val_samples = len(X) // K
all_scores = []

# Iterate through each fold
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

    if classifier == 0:  # K-Nearest Neighbors
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)
        acc = accuracy_score(prediction, test_label)

    if classifier == 1:  # Decision Tree
        model = DecisionTreeClassifier()
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)
        acc = accuracy_score(prediction, test_label)

    if classifier == 2:  # Random Forest
        model = RandomForestClassifier()
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)
        acc = accuracy_score(prediction, test_label)

    if classifier == 3:  # Support Vector Machine
        model = svm.SVC(kernel="rbf")
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)
        acc = accuracy_score(prediction, test_label)

    if classifier == 4:  # Artificial Neural Network
        model = build_ann(train_data.shape[1])
        model.fit(train_data, train_label, epochs=1000, batch_size=256, verbose=0)
        _, acc = model.evaluate(test_data, test_label, verbose=0)

    print(f"Fold {k} ACC: {acc:.4f}")
    all_scores.append(acc)

# Final evaluation summary
print(
    f"### SR - {diag_list[G]} | {entropy_names[entropy_id]} | Classifier: {classifier_list[classifier]} ###"
)
print(
    "ACC= %.2f%% +- %.2f%%"
    % (
        np.mean(all_scores) * 100,
        np.std(all_scores) * 100,
    )
)
