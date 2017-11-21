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

#### 1.2 Data Preprocessing

All the data provided for the acceleration and the rotation rate used the coordinate system from the cellphone as the reference. This coordinate system is pictured below:

| ![Cellphone's coordinate frame](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Extra/Figures/Dimension_Reduction/Figures/SmartPhoneSensors.gif "Cellphone's coordinate frame") |
|:----:|
| Cellphone's coordinate frame |

In order to facilitate or to obtain a more meaningful feature extraction from the acceleration and the rotation rate, it was applied a rotation using the [quaternion](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/quaternion.py) representation of the gyroscope data. This rotation changed the coordinate system from the phone's frame to the world's frame, defining a unique system across all the data samples.

One common problem in sensor data is the existence of noise, so to get a smoother signal, it was introduced to the pipeline the option to use wavelet for filtering. The most common discrete wavelet families were attempted and the ones that better in each family given the length limitations of the time-series are listed in the table below:

| Wavelet Family | Wavelet code in the PyWavelets package | Level |
|-------------|-------------|-------------|
Haar (haar)	| haar | 3
Daubechies (db)	| db9 | 4
Symlets (sym) | sym9 | 4
Coiflets (coif) | coif3 | 4
Biorthogonal (bior) | bior6.8 | 4
Reverse biorthogonal (rbio) | rbio6.8 | 4

#### 1.3 Feature Generation

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

#### 1.4 Parallelization

Given the number of data samples (34,631), the length of each time-series (up to 3603) for x, y, z components from the rotation rate and the acceleration in the stages considered (walking outbound and rest) and the heavy computing necessary to perform the cleaning, the preprocessing and the feature generation, a parallelized implementation is used in all steps described above. The number of processes created for the parallelization is specified by the user in the [run code](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/run.py). This parallelization was also important to make an efficient use of the batch scheduled HTC cluster used in this work, the NOTS at the Center of Research Computing from Rice University.

#### 1.5 Code structure

The code responsible for the feature generation is divided into 3 parts:
* [cleanFeaturise](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/cleanFeaturise.py): Accesses the CSV files, selects the columns of interest, performs the merge operation and calls the execution of the feature generation.
* [createFeatures](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/createFeatures.py): the Main overview of all the features generated for each sample.
* [features_utils](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/features_utils.py): Collection of functions used to perform the specific operations for each feature.
* [quaternion](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/quaternion.py): Quaternion class with the operations to perform the rotation.
* [utils](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/utils.py): Utility functions used for file manipulation or data preprocessing.
* [run](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/run.py): Top-level script to execute the feature generation starting from the cleaning/preprocessing.

### 2 Machine Learning Modeling

#### 2.1. Random Forest

The first attempt to apply the random forest algorithm to the features listed above had a ROC score at the validation set above 90% even before any hyperparameter tuning. However, this result was outstanding enough to cause skepticism about the validity of the metric used. After a reanalysis of the procedure employed, a data leakage was discovered. Even though the dataset was divided into training and test and a cross-validation method was being employed for hyperparameter tuning, the fact that the same person could participate in the experiment multiple times created an intersection between the set that led to incoherent results. This problem was solved by taking account of the healthcode during the creation of the sets and by avoiding the use of cross-validation with the creation of a validation set.

After applying the solution to the data leakage, the ROC score dropped to about 75% on the validation set even after hyperparameter tuning, while the training ROC score remained above 90% which suggested overfitting. With this 		((((CONTINUE))))

#### 2.2 RNN

[Deep Learning for Time-Series Analysis](https://arxiv.org/abs/1701.01887#)

[Time series classification with Tensorflow](https://burakhimmetoglu.com/2017/08/22/time-series-classification-with-tensorflow/)

### 2.3 CNN

[GuanLab's solution to the 2017 Parkinson's Disease Digital Biomarker DREAM Challenge](https://www.synapse.org/#!Synapse:syn10146135/wiki/448409)

[Automated Extraction of Digital Biomarkers using A Hierarchy of Convolutional Recurrent Attentive Neural Networks](https://www.synapse.org/#!Synapse:syn10922704/wiki/471154)

#### 2.4 Unsupervised Learning

[Time Series Classification and Clustering with Python](http://alexminnaar.com/time-series-classification-and-clustering-with-python.html)

[k-Shape](http://www1.cs.columbia.edu/~jopa/Papers/PaparrizosSIGMOD2015.pdf)

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

### 4 TODO

- [ ] Completing README
----------------


>This work was supported in part by the Big-Data Private-Cloud Research Cyberinfrastructure MRI-award funded by NSF under grant CNS-1338099 and by Rice University.