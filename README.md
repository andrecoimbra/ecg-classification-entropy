# ECG_Signal_Classification_entropy_of_recurrence_microstates
 This supplemental material provides the code implementation used in the study for ECG signal classification.

Our study addresses the classification of cardiac arrhythmias, presenting a novel approach based on dynamical system techniques, specifically recurrence entropy of microstates and recurrence vicinity threshold, coupled with artificial intelligence. Leveraging a comprehensive 12-lead electrocardiogram open dataset with over 10,000 subjects and 11 distinct heart rhythms, our work stands out for its significant reduction in dataset dimensions (from 12x5000 to 12x2), enhancing the efficiency of machine learning algorithms for rapid and accurate analyses.

We introduce a unique application of recurrence-based tools, highlighting the automatic maximization of the entropy of recurrence microstates, coupled with the vicinity threshold, as a potent feature-extractor for distinct time series. Our methodology demonstrates robust results in binary-classification setups, achieving an accuracy of 97.5% in distinguishing Sinus Rhythm (SR) from Supraventricular Tachycardia (SVT) and an overall accuracy of 79.45% in multiclass-classification setups, showcasing its general ability for feature-extraction.

Based on only two quantifiers, microstate entropy and recurrence threshold, our results open avenues for future exploration. We emphasize the potential improvement by incorporating additional quantifiers, such as numerical values of recurrence microstate probabilities, to enhance the discernment of patterns by machine learning algorithms.

*To ensure proper attribution, It would be greatly appreciated if you could kindly cite our paper <code>DOI:XXXXXX</code>.*<br />
*Thank you for considering our work!*

## Dataset

- The dataset is open an freely available and can be downloaded in https://figshare.com/collections/ChapmanECG/4560497/2;
- The dataset is composed of 12-lead electrocardiogram signals of 10,646 subjects including $5,956$ males and 4,690 females;
- Among all patients, 17% had normal sinus rhythm, and 83% had at least one abnormality;
- The data of each subject were acquired with a 12-lead resting ECG test that was taken over 10 seconds with a sampling rate of 500 Hz, resulting in 5000 data points per lead per subject;
-  A detailed description of the distribution of the rhythm frequency of the subjects is presented in the Table below:

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


## Downloading the data:
- Download the <code>ECGDataDenoised.zip</code> from https://figshare.com/collections/ChapmanECG/4560497/2;
- Extract the <code>.zip</code> file to obtain the folder <code>ECGDataDenoised.</code> which contains the whole dataset in <code>csv</code> form; 
- Download <code>filename.dat</code> from this repository which contains the list of names of each file from each subject;
- Download <code>diagnostics.dat</code> from this repository which contains the list the ECG rhythms;
- Download <code>block_list.dat</code> from this repository which contains the list corrupted signals that should be avoided;



## Python libraries:

Keras is a high-level deep learning framework designed for rapid experimentation and prototyping of neural networks. It offers an intuitive interface for building, training, and deploying models, with modular components known as layers that can be easily configured to create complex architectures. Keras seamlessly integrates with popular deep learning backends like TensorFlow, enabling efficient computation on both CPU and GPU. Its user-friendly API abstracts away low-level implementation details, allowing researchers and practitioners to focus on model design and experimentation



