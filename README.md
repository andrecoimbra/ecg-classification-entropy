# The use of entropy of recurrence microstates and artificial intelligence to detect cardiac arrhythmia in ECG records

This repository provides the code implemented in the use of entropy of recurrence microstates and artificial intelligence to detect cardiac
arrhythmia in ECG records

Our study addresses the classification of cardiac arrhythmias, presenting a novel approach based on dynamical system techniques, specifically recurrence entropy of microstates and recurrence vicinity threshold, coupled with artificial intelligence. Leveraging a comprehensive 12-lead electrocardiogram open dataset with over 10,000 subjects and 11 distinct heart rhythms, our work stands out for its significant reduction in dataset dimensions (from 12x5000 to 12x2), enhancing the efficiency of machine learning algorithms for rapid and accurate analyses.

We introduce a unique application of recurrence-based tools, highlighting the automatic maximization of the entropy of recurrence microstates, coupled with the vicinity threshold, as a potent feature-extractor for distinct time series. Our methodology demonstrates robust results in binary-classification setups, achieving an accuracy of 97.5% in distinguishing Sinus Rhythm (SR) from Supraventricular Tachycardia (SVT) and an overall accuracy of 79.45% in multiclass-classification setups, showcasing its general ability for feature-extraction.

Based on only two quantifiers, microstate entropy and recurrence threshold, our results open avenues for future exploration. We emphasize the potential improvement by incorporating additional quantifiers, such as numerical values of recurrence microstate probabilities, to enhance the discernment of patterns by machine learning algorithms.

To ensure proper attribution, it would be greatly appreciated if you could kindly cite our paper <code>DOI:XXXXXX</code>.<br />
Thank you for considering our work!

## Dataset

- The dataset is open and freely available for download at https://figshare.com/collections/ChapmanECG/4560497/2;
- The dataset comprises 12-lead electrocardiogram signals from 10,646 subjects, including 5,956 males and 4,690 females;
- Among all patients, 17% exhibited normal sinus rhythm, while 83% displayed at least one abnormality;
- Data for each subject were acquired through a 12-lead resting ECG test lasting 10 seconds, with a sampling rate of 500 Hz, resulting in 5,000 data points per lead per subject;
- A detailed description of the distribution of rhythm frequency among the subjects is presented in the table below:

| Rhythms Name                            | Acronym Name | Subjects (\%) |
|-----------------------------------------|--------------|---------------|
| Sinus Rhythm                            | SR           | 1,826 (17.15) |
| Sinus Bradycardia                       | SB           | 3,889 (36.53) |
| Atrial Fibrillation                     | AFIB         | 1,780 (16.72) |
| Sinus Tachycardia                       | ST           | 1,568 (14.73) |
| Supraventricular Tachycardia            | SVT          | 587 (5.51)    |
| Atrial Flutter                          | AF           | 445 (4.18)    |
| Sinus Irregularity                      | SI           | 399 (3.75)    |
| Atrial Tachycardia                      | AT           | 121 (1.14)    |
| Atrioventricular Node Reentrant Tachycardia | AVNRT    | 16 (0.15)     |
| Atrioventricular Reentrant Tachycardia  | AVRT         | 8 (0.07)      |
| Sinus Atrium to Atrial Wandering Rhythm | SAAWR        | 7 (0.07)      |
| **Total**                               |              | **10,646 (100)** |

## Python libraries:

- **NumPy**: Facilitates efficient handling and manipulation of large multi-dimensional arrays and provides a wide range of mathematical functions for numerical computations in Python;
- **sys**: The sys module provides access to some variables used or maintained by the Python interpreter and functions that interact strongly with the interpreter;
- **csv**: Used to read the .csv files;
- **TensorFlow**: TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive ecosystem of tools, libraries, and community resources for building and deploying machine learning models across a range of platforms.
- **scikit-learn**: Scikit-learn is a simple and efficient tool for data mining and data analysis. It provides a wide range of supervised and unsupervised learning algorithms through a consistent interface in Python.
- **Keras**: is a high-level deep learning framework designed for rapid experimentation and prototyping of neural networks. It offers an intuitive interface for building, training, and deploying models, with modular components known as layers that can be easily configured to create complex architectures. Keras seamlessly integrates with popular deep learning backends like TensorFlow, enabling efficient computation on both CPU and GPU. Its user-friendly API abstracts away low-level implementation details, allowing researchers and practitioners to focus on model design and experimentation.

## Downloading the data:
- Download the <code>ECGDataDenoised.zip</code> from https://figshare.com/collections/ChapmanECG/4560497/2;
-  Extract the <code>.zip</code> file to obtain the folder <code>ECGDataDenoised/</code>, which contains the entire dataset in <code>csv</code> format;
- Download <code>filename.dat</code> from this repository, which contains the list of filenames for each subject;
- Download <code>diagnostics.dat</code> from this repository, which contains the list of ECG rhythms;
- Download <code>block_list.dat</code> from this repository, which contains the list of corrupted signals that should be avoided.<br />
*Please note that when utilizing this step, it is important to cite the article <code>DOI:10.1038/s41597-020-0386-x</code> for proper acknowledgment.*

## Evaluating the Maximum Entropy of Microstates
- The file <code>evaluate_entropy_ECG.py</code> available in this repository contains the code to read the dataset and evaluate the Maximum Entropy of Microstates for each subject;
- The time required to calculate the maximum entropy of microstates depends on the number of microstates extracted from the series;
- In general, entropy calculation is fast, but when considering a dataset with 10,000 subjects containing 12 series of 5,000 points per subject, it can take a relatively long time, requiring a few seconds per subject. Therefore, to address this issue, it is recommended to utilize parallel computing on CPU or GPU;
- The code extracts 1,000 microstates from each time series, with N = 3 resulting in a total of Q = 512 different recurrence microstates. It generates two <code>.npy</code> files: <code>Data_S.npy</code>, containing 12 values of maximum entropy and the 12 values of optimized threshold for each subject, and <code>Data_L.npy</code>, containing the list of labels ranging from 0 (SR) to 10 (SAAWR), following the order of the table above.<br />
*Please note that when utilizing this step, it is important to cite the article <code>DOI:10.1063/1.5125921</code> for proper acknowledgment.*

In case you don't want to wait, you can download <code>10000_Data_S.npy</code> and <code>10000_Data_L.npy</code> from this repository, which correspond to the output of the code considering 10,000 recurrence microstates extracted from each subject.

## About the multiclass code:

- The file <code>multi_class_reduced.py</code> available in this repository contains the code for reading the <code>.npy</code> files and classifying the ECG signals from the reduced dataset.
- The code is divided into 6 steps:
  1. Read <code>Data_S.npy</code> and <code>Data_L.npy</code>, which contain the features and labels of the dataset;
  2. Select the groups of interest, such as the rhythms for the reduced dataset, which includes subjects with rhythms 0, 1, 2, and 3;
  3. Shuffle the dataset to avoid any bias in the disposition of patients;
  4. Define the classifier and its properties;
  5. Generate the K-fold (with K=10) to select the training and test datasets for evaluating the classification;
  6. The code prints the mean value of all K realizations on the screen.

## Instructions for running the multiclass code:
- The command <code>python3 multi_class_reduced.py X</code> runs the code described above, where X is a number from 0 to 4 that selects the classifier to be used (see the labels below). If no flag <code>X</code> is provided, the code will automatically select the Artificial Neural Network as the classifier.
- If you wish to use the <code>10000_Data_S.npy</code> and <code>10000_Data_L.npy</code> files, you should change line #16 from <code>data_in_S, data_in_L = np.load('Data_S.npy'), np.load('Data_L.npy')</code> to <code>data_in_S, data_in_L = np.load('10000_Data_S.npy'), np.load('10000_Data_L.npy')</code>.

## About the binary classification code:

- The file <code>binary_class.py</code> available in this repository contains the code for reading the <code>.npy</code> files and classifying the ECG signals.
- The code works similarly to the multiclass case, but it was adapted to perform a binary classification;
- The command <code>python3 binary_class.py G X</code> runs the code described above, where G is a number from 1 to 10, which selects the ECG rhythms to be compared with the SR, and X is a number from 0 to 4 that selects the classifier to be used (see the labels below). If no flag <code>X</code> is provided, the code will automatically select the Artificial Neural Network as the classifier.

## About the classifiers:

In this work, we tested 5 different classifiers, which are:
- KNN: K-Nearest Neighbors (label 0) using the function <code>KNeighborsClassifier(n_neighbors=10)</code>, where <code>10</code> represents the number of nearest neighbors to be analyzed.
- DTree: Decision Tree (label 1) using the function <code>DecisionTreeClassifier()</code>.
- RFM: Random Forest (label 2) using the function <code>RandomForestClassifier()</code>.
- SVM: Support Vector Machine (label 3) using the function <code>svm.SVC(kernel='rbf')</code>, where <code>rbf</code> represents the Radial Basis Function (RBF) kernel of the <code>SVM</code>.
- ANN: Artificial Neural Network (label 4) using the Keras framework to generate a feedforward network designed for classification tasks. It has an input layer with 24 nodes, two hidden layers with 64 nodes each and ReLU activation functions, a dropout layer with a dropout rate of 0.5, and an output layer. 

## Final Remarks

In conclusion, this repository offers a comprehensive implementation of entropy of recurrence microstates combined with artificial intelligence for cardiac arrhythmia detection in ECG records. The provided code facilitates experimentation and exploration in this vital area of medical research, aiming to contribute to the advancement of cardiac health diagnostics and treatment.

We encourage collaboration and feedback from the scientific community to foster continued development and refinement of these techniques. Together, we can continue to push the boundaries of knowledge and technology to address the complex challenges in cardiovascular health and pave the way for more effective medical interventions.

If you find this repository helpful, please consider citing our paper with <code>DOI: XXXXXX</code> to acknowledge our contributions.




  






