  # ECG_Signal_Classification_entropy_of_recurrence_microstates
  
This repository provides the code implemented in the project entitled: "The Use of Entropy of Recurrence Microstates and Artificial Intelligence to Detect Cardiac Arrhythmia in ECG Records."

Our study addresses the classification of cardiac arrhythmias, presenting a novel approach based on dynamical system techniques, specifically the entropy of recurrence microstates and recurrence vicinity threshold, coupled with artificial intelligence. Leveraging a comprehensive 12-lead electrocardiogram open dataset with over 10,000 subjects and 11 distinct heart rhythms, our work stands out for its significant reduction in dataset dimensions (from 12x5000 to 12x2), enhancing the efficiency of machine learning algorithms for rapid and accurate analyses.

We introduce a unique application of recurrence-based tools, highlighting the automatic maximization of the entropy of recurrence microstates, coupled with the vicinity threshold, as a potent feature extractor for distinct time series. Our methodology demonstrates robust results in binary classification setups, achieving an accuracy of 97.5% in distinguishing Sinus Rhythm (SR) from Supraventricular Tachycardia (SVT) and an overall accuracy of 79.45% in multiclass classification setups, showcasing its general ability for feature extraction.

Based on only two quantifiers, microstate entropy and recurrence threshold, our results open avenues for future exploration. We emphasize the potential improvement by incorporating additional quantifiers, such as numerical values of recurrence microstate probabilities, to enhance the discernment of patterns by machine learning algorithms.
  
<!--To ensure proper attribution, it would be greatly appreciated if you could kindly cite our paper <code>DOI:XXXXXX</code>.<br />
  Thank you for considering our work!-->

## Repository Contents

This repository contains: 

- **Datasets:** Instructions to download the experimental ECG dataset utilized in the analysis;
- **Codes:** Implementation of entropy of recurrence microstates and machine learning algorithms as described in the paper. </br>
With the provided resources, users can replicate the majority of the results presented in the manuscript.

## Dataset

- The ECG signal dataset is openly accessible for download at https://figshare.com/collections/ChapmanECG/4560497/2;
- The dataset includes 12-lead electrocardiogram signals collected from 10,646 subjects, comprising 5,956 males and 4,690 females;
- Among all subjects, 17% exhibited normal sinus rhythm, while 83% displayed at least one abnormality;
- Each subject's data were obtained from a 12-lead resting ECG test lasting 10 seconds, with a sampling rate of 500 Hz, resulting in 5,000 data points per lead per subject;
- A comprehensive breakdown of rhythm frequency distribution among the subjects is provided in the table below:

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

- **NumPy**: Facilitates efficient handling and manipulation of large multi-dimensional arrays and provides a wide range of mathematical functions for numerical computations in Python.
- **sys**: The sys module provides access to some variables used or maintained by the Python interpreter and functions that interact strongly with the interpreter.
- **csv**: Used to read the .csv files.
- **TensorFlow**: An open-source machine learning framework developed by Google, TensorFlow offers a comprehensive ecosystem of tools, libraries, and community resources for building and deploying machine learning models across various platforms.
- **scikit-learn**: A simple and efficient tool for data mining and analysis, scikit-learn provides a wide range of supervised and unsupervised learning algorithms through a consistent interface in Python.
- **Keras**: A high-level deep learning framework designed for rapid experimentation and prototyping of neural networks. Keras offers an intuitive interface for building, training, and deploying models, with modular components known as layers that can be easily configured to create complex architectures. It seamlessly integrates with popular deep learning backends like TensorFlow, enabling efficient computation on both CPU and GPU. Its user-friendly API abstracts away low-level implementation details, allowing researchers and practitioners to focus on model design and experimentation.

## Downloading the data:
- Download the `ECGDataDenoised.zip` from [here](https://figshare.com/collections/ChapmanECG/4560497/2).
- Extract the `.zip` file to obtain the folder `ECGDataDenoised/`, which contains the entire dataset in `csv` format.
- Download `filename.dat` from this repository, which contains the list of filenames for each subject.
- Download `diagnostics.dat` from this repository, which contains the list of ECG rhythms.
- Download `block_list.dat` from this repository, which contains the list of corrupted signals that should be avoided.  
*Please remember to cite the article `DOI:10.1038/s41597-020-0386-x` for proper acknowledgment when utilizing this step.*

## Evaluating the Maximum Entropy of Microstates (optional)
- The file `evaluate_entropy_ECG.py` available in this repository contains the code to read the dataset and evaluate the Maximum Entropy of Microstates for each subject.
- The time required to calculate the maximum entropy of microstates depends on the number of microstates extracted from the series.
- In general, entropy calculation is fast, but when considering a dataset with 10,000 subjects containing 12 series of 5,000 points per subject, it can take a relatively long time, requiring a few seconds per subject. Therefore, to address this issue, it is recommended to utilize parallel computing on CPU or GPU.
- The code extracts 1,000 microstates from each time series, with \( N = 3 \) resulting in a total of \( Q = 512 \) different recurrence microstates. It generates two `.npy` files: `Data_S.npy`, containing 12 values of maximum entropy and the 12 values of the optimized threshold for each subject, and `Data_L.npy`, containing the list of labels ranging from 0 (SR) to 10 (SAAWR), following the order of the table above.  
*Please remember to cite the article `DOI:10.1063/1.5125921` for proper acknowledgment when utilizing this step.*

In case you don't want to wait, you can download `10000_Data_S.npy` and `10000_Data_L.npy` from this repository, which correspond to the output of the code considering 10,000 recurrence microstates extracted from each subject.

## About the multiclass code:

- The file `multi_class_reduced.py` available in this repository contains the code for reading the `.npy` files and classifying the ECG signals from the reduced dataset.
- The code is structured into 6 steps:
  1. Reading `Data_S.npy` and `Data_L.npy`, which contain the features and labels of the dataset;
  2. Selecting the groups of interest, such as the rhythms for the reduced dataset, which includes subjects with rhythms 0, 1, 2, and 3;
  3. Shuffling the dataset to avoid any bias in the disposition of patients;
  4. Defining the classifier and its properties;
  5. Generating the K-fold (with K=10) to select the training and test datasets for evaluating the classification;
  6. Print the mean value of all K realizations on the screen.

## Instructions for running the multiclass code:
- To run the code described above, execute the command `python3 multi_class_reduced.py argv_1`, where `argv_1` is a number from 0 to 4 representing the selected classifier (refer to the labels below). If no flag `argv_1` is provided, the code will automatically choose the Artificial Neural Network as the classifier.
- If you prefer to use the `10000_Data_S.npy` and `10000_Data_L.npy` files, modify line #16 from `data_in_S, data_in_L = np.load('Data_S.npy'), np.load('Data_L.npy')` to `data_in_S, data_in_L = np.load('10000_Data_S.npy'), np.load('10000_Data_L.npy')`.

## About the binary classification code:

- The file <code>binary_class.py</code> available in this repository contains the code for reading the <code>.npy</code> files and classifying the ECG signals.
- The code works similarly to the multiclass case, but it was adapted to perform a binary classification;
- The command <code>python3 binary_class.py argv_1 argv_2</code> runs the code described above, where `argv_1` is a number from 1 to 10, which selects the ECG rhythms to be compared with the SR, and `argv_2` is a number from 0 to 4 that selects the classifier to be used (see the labels below). If no flag <code>argv_2</code> is provided, the code will automatically select the Artificial Neural Network as the classifier.

## About the classifiers:

In this study, we evaluated five different classifiers, as follows:
- KNN: K-Nearest Neighbors (label 0) implemented using the function <code>KNeighborsClassifier(n_neighbors=10)</code>, where <code>10</code> denotes the number of nearest neighbors to be considered.
- DTree: Decision Tree (label 1) implemented using the function <code>DecisionTreeClassifier()</code>.
- RFM: Random Forest (label 2) implemented using the function <code>RandomForestClassifier()</code>.
- SVM: Support Vector Machine (label 3) implemented using the function <code>svm.SVC(kernel='rbf')</code>, where <code>rbf</code> represents the Radial Basis Function (RBF) kernel of the <code>SVM</code>.
- ANN: Artificial Neural Network (label 4) implemented using the Keras framework to construct a feedforward network tailored for classification tasks. It consists of an input layer with 24 nodes, two hidden layers with 64 nodes each, employing ReLU activation functions, a dropout layer with a dropout rate of 0.5, and an output layer.

## Final Remarks

In conclusion, this repository presents a comprehensive implementation of the entropy of recurrence microstates combined with artificial intelligence for the detection of cardiac arrhythmias in ECG records. By providing access to the code, we aim to facilitate experimentation and exploration in this critical area of medical research, with the ultimate goal of advancing diagnostics and treatment for cardiac health.

We welcome collaboration and feedback from the scientific community to foster ongoing development and refinement of these techniques. Together, we can continue to push the boundaries of knowledge and technology to address the complex challenges in cardiovascular health and pave the way for more effective medical interventions.

### Citation

If you find this work beneficial for your research, we kindly encourage you to consider reading and citing:

- Boaretto, B. R. R., Andreani, A. Lopes, S. R., Prado, T. L., & Macau, E. E. N. (2024). "The use of entropy of recurrence microstates and artificial intelligence to detect cardiac arrhythmia in ECG records." Under Review.

Additionally, the methodology utilized in this work has been developed and refined in the following publications:
- Corso, G., Prado, T. D. L., Lima, G. Z. D. S., Kurths, J., & Lopes, S. R. (2018). "Quantifying entropy using recurrence matrix microstates." *Chaos: An Interdisciplinary Journal of Nonlinear Science*, 28(8).
- Prado, T. L., Corso, G., dos Santos Lima, G. Z., Budzinski, R. C., Boaretto, B. R., Ferrari, F. A. S., Macau, E. E. N., & Lopes, S. R. (2020). "Maximum entropy principle in recurrence plot analysis on stochastic and chaotic systems." *Chaos: An Interdisciplinary Journal of Nonlinear Science*, 30(4).
- Prado, T. L., Boaretto, B. R. R., Corso, G., dos Santos Lima, G. Z., Kurths, J., & Lopes, S. R. (2022). "A direct method to detect deterministic and stochastic properties of data." *New Journal of Physics*, 24(3), 033027.

Thank you for your interest in our research! We hope this repository serves as a valuable resource for utilizing entropy of recurrence microstates in signal classification and data analysis.

Sincerely,</br>
Bruno R. R. Boaretto.
  






