# Parkinson's Disease Digital Biomarker

#### Study done with the dataset of the Parkinson’s Disease Digital Biomarker DREAM Challenge from the website http://dreamchallenges.org/ .

------------
## The Data

The data is a collection of time-series from cellphone sensors (pedometer at an undefined sampling rate and the following three at a rate of 100Hz: accelerometer, gyroscope, and magnetometer) of the gait and balance of people with or without Parkinson's disease. In each sample, the person is instructed to put the cellphone in a pant's pocket and perform measurements in three stages: 20 steps in a straight line (walking outbound); turn around and stand still for 30 seconds (walking rest); 20 steps back in a straight line (walking return). There is also a collection of demographics data for each person who has participated in the experiment.

Each time-series is stored in a different JSON file which is referenced by a pointer (file code) in a CSV file ("walking_activity.csv") that contains one row for each sequence of the three measurements. Each row from this CSV file also has a healthcode which is a unique code for each person that can be used to associate the data from the time-series with the data of another CSV file ("demographics.csv") that contains one row with all demographics data for each person.

###### Example of the json structure for each data sample collected at a rate of 100Hz:
```json
{
	"attitude":{
		"y":0.07400703039259371,
		"w":0.6995846050657238,
		"z":-0.002046270534428376,
		"x":0.710703983796633
	},
	"timestamp":
		136377.071964625,
	"rotationRate":{
		"x":0.723730206489563,
		"y":-1.446872472763062,
		"z":0.3764457404613495
	},
	"userAcceleration":{
		"x":-0.003811069531366229,
		"y":0.05102279782295227,
		"z":-0.003759366692975163
	},
	"gravity":{
		"x":0.1064569428563118,
		"y":-0.9940922260284424,
		"z":0.02115438692271709
	},
	"magneticField":{
		"y":0,
		"z":0,
		"x":0,
		"accuracy":-1
	}
}
```

More details about parameter descriptions at https://www.synapse.org/#!Synapse:syn8717496/wiki/448355 .

-----------
## Sequence of studies and uses of the dataset

### 1 Before the Machine Learning Models

#### 1.1 Cleaning

The first part was to remove clear cases of inconsistency in the data. There were cases of invalid references to JSON files or JSON files with 'null' value. Only this cleaning procedure resulted in a significant reduction in the data size, from more than 34,000 samples to less than 20,000. However, it was later observed that the vast majority of the inconsistent data was from the walking return stage, so the option adopted was to drop all the return data and then apply the cleaning procedures. 

The time-series with a length smaller than 300 samples also were also discarded. Those correspond to 3 seconds of data collection each and this duration was considered unreasonable to execute 20 steps for this data collection purpose or insufficient for the resting stage which was supposed to last for 30 seconds.

All the code related to the cleaning procedures above are part of the function [rowFeaturise](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/cleanFeaturise.py) that retrieves the data from the JSON file before starting the feature generation described in the next topic.

It's also worth mentioning that most of the demographics data were incomplete, so only the age, the gender and professional diagnosis (target) were kept.

After executing all the cleaning procedures described above, 2,225 samples from a total of 34,631 were dropped.

Extra steps of data cleaning are also performed while [splitting the dataset](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/splitSets.py) into training, validation and test sets. The first one was to remove the inconsistency between the label and medicine timepoint. There were cases in which the label was negative for the Parkinson's Disease and the person would still take the medicine. Besides that, it was opted to discard the data of people with Parkinson's disease that did the experiment right after taking the medicine since the idea of the model after deployed is to predict Parkinson's disease among people without the diagnosis, therefore not taking the medication, and the use of the drug could cause a behavior of a healthy person. The table below associate the dropped data with the medicine timepoint label:

| medTimepoint | professional-diagnosis | Number of samples | Dropped?
|------------|------------|------------|------------| 
Another time | False <br/> True | 43 <br/> 10789 | Yes <br/> No
I don't take Parkinson medications | False <br/> True | 9690 <br/> 1494 | No <br/> No
Immediately before Parkinson medication | False <br/> True | 33	<br/> 5445 | Yes <br/> No
Just after Parkinson medication (at your best) | False <br/> True | 19 <br/> 5017 | Yes <br/> Yes

The second step of data cleaning that is performed in the splitting procedure aims to solve the skewed distribution of samples per healthCode in the dataset. There were people that recorded up to 450 samples of data while the average number of samples per healthCode is around 11. This asymmetrical distribution could have negative effects on the training data, possibly causing a slight overfitting for a specific healthcode which would be especially bad if a specific individual corresponds to an outlier. The implemented solution to solve this issue was to limit the number of samples per person, using only the first 10 occurrences per healthCode. However, this solution implies a huge loss in the dataset, so it is optionally executed as set by a boolean parameter in the function [generateSetTables](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/splitSets.py).

#### 1.2 Data Preprocessing

All the data provided for the acceleration and the rotation rate used the coordinate system from the cellphone as the reference. This coordinate system is pictured below:

| ![Cellphone's coordinate frame](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Extra/Figures/SmartPhoneSensors.gif "Cellphone's coordinate frame") |
|:----:|
| Cellphone's coordinate frame |

In order to facilitate or to obtain a more meaningful feature extraction from the acceleration and the rotation rate, it was applied a rotation using the [quaternion](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/quaternion.py) representation of the gyroscope data. This rotation changed the coordinate system from the phone's frame to the world's frame, defining a system that is consistent across all the measurements within each experiment.

One common problem in sensor data is the existence of noise, so to get a smoother signal, it was introduced to the pipeline the option to use wavelet for filtering. The most common discrete wavelet families were attempted and the ones that better in each family given the length limitations of the time-series are listed in the table below:

| Wavelet Family | Wavelet code in the PyWavelets package | Level |
|-------------|-------------|-------------|
Haar (haar)	| haar | 3
Daubechies (db)	| db9 | 4
Symlets (sym) | sym9 | 4
Coiflets (coif) | coif3 | 4
Biorthogonal (bior) | bior6.8 | 4
Reverse biorthogonal (rbio) | rbio6.8 | 4

#### 1.3 Data Augmentation

Following the insights of the [GuanLab's solution](https://www.synapse.org/#!Synapse:syn10146135/wiki/448409), two steps of data augmentation were implemented in order to enrich the training data:

* The first step was the simulation of people walking faster or slower, this was done by the use of a random multiplicative factor in the interval [0.8, 1.2) which would account for tremors at different frequencies.

* The second type of data augmentation was the use of a random 3D rotation to simulate people holding the phone in different orientations which would correspond to different world frame coordinate systems. However, it is important to highlight that the rotation is limited to the plane orthogonal to the z-axis since this axis is always well defined by the gravity in the world's frame. It is also worth mentioning that this rotation is not contradictory with the rotation executed to convert the data from the phone's frame to the world's frame. This is explained by the fact that the rotation there was based in the quaternions from the gyroscope data which changes for each timestamp while the augmentation uses a single quaternion, applying the same corresponding rotation to all the samples in a 3D time series.

All the code that performs the data augmentation described above can be found in the [function augmentData](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/utils.py) that receives as parameter the fraction of the training data that is going to be augmented.

#### 1.4 Feature Generation

The first approach was trying to reproduce the results obtained in a similar experiment as reported in the article [High Accuracy Discrimination of Parkinson's Disease Participants from healthy controls using smartphones](http://ieeexplore.ieee.org/document/6854280/). The first step was to generate the features listed in the article:

| Brief Feature Description | Name of the column in the CSV file generated -Suffix[<sup>1</sup>](#foot2.1_1) |
| ------------- |-------------|
μ Mean | Mean 
σ Standard deviation | Std Deviation
25th percentile | Q1
75th percentile | Q3
Inter-quartile range (Q3 − Q1) | Interquartile
Median | Median
~~Mode~~ |
Data range (maximum – minimum) | Data Range
Skewness | Skewness
Kurtosis | Kurtosis
Mean squared energy | Root Mean Square (RMS)
Entropy | Entropy
Zero-crossing rate | Zero Crossing Rate
Dominant frequency component | Dominant Frequency Component
Radial distance | Radial Distance Mean
θ Polar angle | Polar Angle Mean
Azimuth angle | Azimuth Angle Mean
Cross-correlation between the acceleration in two axes | Cross-correlation 0 lag
Mutual information between the acceleration in two axes | Mutual Information
~~Cross-entropy between the acceleration in two axes~~
~~DFA[<sup>2</sup>](#foot2.1_2) (Detrended Fluctuation Analysis) Extent of randomness in body motion~~
~~TKEO (Teager-Kaiser Energy Operator) Instantaneous changes in energy due to body motion~~
~~Autoregression coefficient at time lag 1~~

Other features that are not listed in the article were also added:

| Name of the column in the CSV file generated -Suffix[<sup>1</sup>](#foot2.1_2) |
| -------------------------------------------------|
Signal Magnitude Area (SMA[<sup>3</sup>](#foot2.1_3))
Maximum
Minimum
Distance walked
Average Speed
Average Size of the Step

A possible future improvement is trying to incorporate more features as described in the article: [Preprocessing Techniques for Context Recognition from Accelerometer Data](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.357.3920&rep=rep1&type=pdf).

Other features of interest to be included:
* meanFreq(): Weighted average of the frequency components to obtain a mean frequency
* arCoeff(): Autorregresion coefficients with Burg order equal to 4
* Power Spectral Density

##### Footnotes:
<a name="foot2.1_1"></a>[1] Name termination that specifies which time-series (outbound, rest or return) from the rotation rate or the acceleration and which axis(es) is the feature associated with. <br />
<a name="foot2.1_3"></a>[2] As described in the article [Feature Selection and Activity Recognition System Using a Single Triaxial Accelerometer](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6780615&tag=1). <br />
<a name="foot2.1_3"></a>[3] As described in the article [Activity recognition from acceleration data collected with a tri-axial accelerometer](http://users.utcluj.ro/~ATN/papers/ATN_2_2011_6.pdf). <br />

#### 1.5 Parallelization

Given the number of data samples (34,631), the length of each time-series (up to 3603) for x, y, z components from the rotation rate and the acceleration in the stages considered (walking outbound and rest) and the heavy computing necessary to perform the cleaning, the preprocessing and the feature generation, a parallelized implementation is used in all steps described above. The number of processes created for the parallelization is specified by the user in the [run code](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/run.py). This parallelization was also important to make an efficient use of the batch scheduled HTC cluster used in this work, the NOTS at the Center of Research Computing from Rice University. However, it's worth highlighting that even with the creation of 16 processes on the NOTS system, the total run time is around 5 hours and 25 minutes.

#### 1.6 Code structure

The code responsible for the feature generation is divided into 3 parts:
* [cleanFeaturise](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/cleanFeaturise.py): Accesses the walking activity CSV table, executes the first stages described for data cleaning, performs the feature generation and saves the results in a new table.
* [splitSets](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/splitSets.py): Accesses the demographics table and also the table generated by the cleanFeaturise code, performs the merge operation taking the healthCode in consideration and also executes the final steps of data cleaning.
* [createFeatures](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/createFeatures.py): the Main overview of all the features generated for each sample.
* [features_utils](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/features_utils.py): Collection of functions used to perform the specific operations for each feature.
* [quaternion](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/quaternion.py): Quaternion class with the operations to perform the rotation.
* [utils](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/utils.py): Utility functions used for file manipulation or data preprocessing.
* [run](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/run.py): Top-level script to execute the cleanFeaturise code or the data augmentation.

### 2 Machine Learning Modeling

#### 2.1. Random Forest
//NOT THE FINAL VERSION

The first attempt to apply the random forest algorithm to the features listed above had a ROC score at the validation set above 90% even before any hyperparameter tuning. However, this result was outstanding enough to cause skepticism about the validity of the metric used. After a reanalysis of the procedure employed, a data leakage was discovered. Even though the dataset was divided into training and test and a cross-validation method was being employed for hyperparameter tuning, the fact that the same person could participate in the experiment multiple times created an intersection between the set that led to incoherent results. This problem was solved by taking account of the healthcode during the creation of the sets and by avoiding the use of cross-validation with the creation of a validation set.

After applying the solution to the data leakage, the ROC score dropped to about 78% on the validation set even after hyperparameter tuning, while the training ROC score remained above 98%. Of course, this model is clearly overfitting, but before applying regularization, it was also observed in the feature importance (the normalized total reduction of the criterion brought by that feature) ranking that the age is by far the most important feature. This result was expected given the nature of the Parkinson's disease, however, it was favoring a model that does not work properly for the prediction of the disease in older people which is the main group of interest. For this reason this reason, it was opted to only consider samples from people older 56 years old in the validation and test sets in the experiments that followed in this model. The addition of this restriction caused a reduction around 31% in the size of the sets.

The immediate impact of this change in the validation set was a big decrease to the ROC score on the validation set to around 0.53, almost random guessing! An improvement to this current performance came from the observation of yet another problem in the dataset: unbalanced label distribution. 19,275 samples from a total of 22,324 samples from people older than 56 years old had a positive diagnosis of Parkinsons Disease. Therefore, the improvement was to balance the training set. The first and simpler approach was undersampling the majority class by randomly selecting samples from the majority class, equating ratio between the two. Undersampling bumped the validation ROC score to the range from 0.56 to 0.63. The Second approach used was oversampling the minority class using the [SMOTE](http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html#from-random-over-sampling-to-smote-and-adasyn) method which increased, even more, the ROC score to the range from 0.60 to 0.67. Given the greater success of the latter, it was chosen in all the next experiments in this model.

#### 2.2 Recurrent Neural Network

[Deep Learning for Time-Series Analysis](https://arxiv.org/abs/1701.01887#)

[Time series classification with Tensorflow](https://burakhimmetoglu.com/2017/08/22/time-series-classification-with-tensorflow/)

#### 2.3 Convolutional Neural Network

This model is based in the [GuanLab's network architecture](https://www.synapse.org/#!Synapse:syn10146135/wiki/448409) in which each 3D time-series from the rotation rate is padded with zeros to a common length of 4000 and used as a one-dimensional image with three channels (one for each axis). The architecture consists of eight one-dimensional convolutional layers intercalated with eight one-dimensional max pooling layers. All the layers do not have padding and all the pooling layers have the same hyperparameter configuration: stride of two and size also equals to two. The convolutional layers have stride one and a monotonically increasing number of filters with hyperparameters described in the table below: 

| Layer Number | Number of <br/> Convolutional Filters | Window Size |
|-------------|-------------|-------------|
1 | 8 | 5
2 | 16 | 5
3 | 32 | 4
4 | 32 | 4
5 | 64 | 4
6 | 64 | 4
7 | 128 | 4
8 | 128 | 5

The output of the final pooling layer is flatenned and used as input to densely-connected layer with linear activation function and two outputs, one logit output for each possible label. All the architecture described so far is illustrated  in the figure below:

| ![Convolutional Neural Network Architecture](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Extra/Figures/CNN_diagram.jpeg "Convolutional Neural Network Architecture") |
|:----:|
| Convolutional Neural Network Architecture |

The code related to the convolutional neural network can be found in the [CNN class](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/CNN/cnn.py) and the part to build the graph of the model described so far is in the logits_prediction method. The structure of the code followed a specific [design recommended for TensorFlow models](https://danijar.com/structuring-your-tensorflow-models/) in order to achieve more maintainable and reusable characteristics.

With the main part of the model done by the logits_prediction method, the logits result is then used in the optimize method in which the probability of each class is calculated with a softmax function and used to calculate the cross-entropy loss function. Finally, the optimize method outputs the gradient descent operation with Adam optimization.

### 3 Extra

#### 3.1. Testing features

With the objective of identifying the right number of bins required by some features, a collection of classes ([testFeature](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/testFeature.py)) that allows the visualization of the value of the feature as a function of the number of bins was created. By looking at the plot generated by these classes and recognizing the number from which the value of the functions achieves a plateau, it's possible to spot the right number of bins to be used in the "official" feature generation.

Besides the definition of parameters for the feature calculation, there is another purpose for some of the classes at [testFeature](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/testFeature.py): data analysis from the visualization of the raw data. To fulfill this purpose, there is the class Overview that generates a grid of plots for all the axes of the accelerometer data and also for the magnitude of the resultant acceleration vector. The Overview class also provides extra options:
* Option to visualize the segments of each step in the graph plotted by making use of the pedometer data, but it lacks precision due to the low quality of the pedometer data;
* Option to apply different types of wavelets for filtering.

It's also worth mentioning that the constructor method of the classes mentioned above requires an interaction with the user for the selection of a set of samples from which a random element will be chosen to be loaded. The user can choose between one of the three stages of the experiment (outbound, rest or return) and if it's from a person with or without Parkinson's disease.

#### 3.2. Features visualization

Due to the high dimensionality of the data even after extracting features (225 dimensions), two approaches were attempted to reduce the number of dimensions in order to provide a clustering visualization that would allow the distinction between time-series from people with and without Parkinson's disease.

The first and simpler method was PCA, to test the viability of this method, it was calculated ([testingPCA](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/Visualization/testingPCA.py)) the variance per axis after fitting the data in a 3-dimensional space. However, the results were very poor (x: 0.22822514, y: 0.0846729, z: 0.07132561) which led to questioning how many dimensions would be enough to achieve a satisfactory variance. Therefore, the data was fitted again but with the variance condition and the result was of 92 dimensions to keep 95% variance, far more than the maximum number for a useful visualization.

To surpass the limitations of a lower dimension projection with PCA, the t-SNE method was applied ([testingTSNE](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/Dimension_Reduction/testingTSNE.py)) and plots for two and three dimensions were generated. However, both figures failed to separate groups data from the two possible targets as can be seen in the pictures below:

| ![2D t-SNE plot](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/Dimension_Reduction/Figures/T-SNE2d.png "2D t-SNE plot") |
|:----:|
| 2D t-SNE plot |


| ![3D t-SNE plot](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/Dimension_Reduction/Figures/T-SNE3d.png "3D t-SNE plot") |
|:----:|
| 3D t-SNE plot |

#### 3.3 Useful Resources

##### 3.3.1 [Understanding ROC curves and Area Under the Curve](https://www.youtube.com/watch?v=OAl6eAyP-yo)

##### 3.3.2 [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

##### 3.3.3 [Hands-On Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do)

##### 3.3.4 [PyWavelets Manual](https://media.readthedocs.org/pdf/pywavelets/latest/pywavelets.pdf)

----------------


>This work was supported in part by the Big-Data Private-Cloud Research Cyberinfrastructure MRI-award funded by NSF under grant CNS-1338099 and by Rice University.