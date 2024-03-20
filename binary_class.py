import numpy as np
import sys 

from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score


#Read Data_S.npy and Data_L.npy which contain the features and labels of the whole dataset;
data_in_S,data_in_L = np.load('Data_S.npy'),np.load('Data_L.npy') #data_in_S, data_in_L = np.load('10000_Data_S.npy'), np.load('10000_Data_L.npy')

#Select the groups of interess 
X = []
Y = []
if len(sys.argv[1]) == 0 or len(sys.argv) <1:
    print('Please, you need to select a group to be compare with the SR rhythms')
else:
    G = int(sys.argv[1])

group_list = np.array([0,G])

diag_list = np.array(['SR','SB','AFIB','ST','SVT','AF','SI','AT','AVNRT','AVRT','SAAWR'])
print('Comparing SR with %s' % diag_list[G])

for i in range(len(data_in_L)):
    if data_in_L[i] in group_list:
        X.append(data_in_S[i])
        if data_in_L[i]>0:
            Y.append(1)
        else:
            Y.append(0)
X = np.array(X)
Y = np.array(Y)

#shuffling the dataset
seed = 1001
np.random.seed(seed)
np.random.shuffle(X)

np.random.seed(seed)
np.random.shuffle(Y)

#Choose the classifier 

if len(sys.argv) > 2:
    classifier = int(sys.argv[2])
else:
    classifier=4
    

classifier_list = ['knn','dtree','rfm','svm','ANN']
#print(classifier_list[classifier])

def build_ann():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(24,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))    
    model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])   
    return model

#K-fold step
K = 10
num_val_samples = len(X) // K
all_scores = []



for k in range(K):
    print('processing fold #', k)
    test_data = X[k * num_val_samples: (k + 1) * num_val_samples]
    test_label = Y[k * num_val_samples: (k + 1) * num_val_samples]

    train_data = np.concatenate([X[:k * num_val_samples],X[(k + 1) * num_val_samples:]],axis=0)
    train_label = np.concatenate([Y[:k * num_val_samples],Y[(k + 1) * num_val_samples:]],axis=0)

    if(classifier==0): #knn        
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)#,test_label)
        acc= accuracy_score(prediction, test_label)

    if(classifier==1): #dtree        
        model = DecisionTreeClassifier()
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)#,test_label)
        acc= accuracy_score(prediction, test_label)

    if(classifier==2): #rfm       
        model = RandomForestClassifier()
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)#,test_label)
        acc= accuracy_score(prediction, test_label) 

    if(classifier==3): #SVM
        model = svm.SVC(kernel='rbf')
        model.fit(train_data, train_label)
        prediction = model.predict(test_data)#,test_label)
        acc= accuracy_score(prediction, test_label)

    if(classifier==4): #ANN
        #train_label = to_categorical(train_label)
        #test_label  = to_categorical(test_label)
        model = build_ann()
        model.fit(train_data, train_label,epochs=1000, batch_size=256,verbose=0)
        _,acc = model.evaluate(test_data,test_label, verbose=0)

    print(acc)
    all_scores.append(acc)
print('Classifier %s with ACC= %.2f +- %.2f' % (classifier_list[classifier],np.mean(all_scores),np.std(all_scores)))

