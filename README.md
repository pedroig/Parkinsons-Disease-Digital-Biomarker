# Parkinson's Disease Digital Biomarker

#### Study done with the dataset of the Parkinson’s Disease Digital Biomarker DREAM Challenge from the website http://dreamchallenges.org/ .

------------
## The Data

The data is a collection of time-series from cellphone sensors (accelerometer at a rate of 100Hz, pedometer, gyroscope, and magnetometer) of the gait and balance of people with or without Parkinson's disease. In each sample, the person is instructed to put the cellphone in a pant's pocket and perform measurements in three stages: 20 steps in a straight line (walking outbound); turn around and stand still for 30 seconds (walking rest); 20 steps back in a straight line (walking return). There is also a collection of demographics data for each person who has participated in the experiment.

Each time-series is stored in a different JSON file which is referenced by a pointer (file code) in a CSV file ("walking_activity.csv") that contains one row for each sequence of the three measurements. Each row from this CSV file also has a healthcode which is a unique code for each person that can be used to associate the data from the time-series with the data of another CSV file ("demographics.csv") that contains one row with all demographics data for each person.

More details about column descriptions at https://www.synapse.org/#!Synapse:syn8717496/wiki/448355 .

-----------
## Sequence of studies and uses of the dataset

### 1. Cleaning

The first part was to remove clear cases of inconsistency in the data. There were cases of invalid references to JSON files or JSON files with 'null' value. Only this cleaning procedure resulted in a significant reduction in the data size, from more than 34,000 samples to less than 20,000. All the code related to this cleaning procedure is part of the function [rowFeaturise](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/cleanFeaturise.py) that retrieves the data from the JSON file before starting the feature generation described in the next topic.

It's also worth mentioning that most of the demographics data were incomplete, so only the age, the gender and professional diagnosis (target) were kept.

### 2. Feature generation and Random Forest

#### 2.1. The features

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

##### Code structure

The code responsible for the feature generation is divided into 3 parts:
* [cleanFeaturise](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/cleanFeaturise.py): Accesses the CSV files, selects the columns of interest, performs the merge operation and calls the execution of the feature generation.
* [createFeatures](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/createFeatures.py): the Main overview of all the features generated for each sample.
* [features_utils](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/features_utils.py): Collection of functions used to perform the specific operations for each feature.

##### Footnotes

<a name="foot2.1_1"></a>[1] Name termination that specifies which time-series (outbound, rest or return) and which axis(es) is the feature associated with.

<a name="foot2.1_3"></a>[2] As described in the article [Feature Selection and Activity Recognition System Using a Single Triaxial Accelerometer](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6780615&tag=1).

<a name="foot2.1_3"></a>[3] As described in the article [Activity recognition from acceleration data collected with a tri-axial accelerometer](http://users.utcluj.ro/~ATN/papers/ATN_2_2011_6.pdf).

#### 2.2. Applying Random Forest

The first attempt to apply the random forest algorithm to the features listed above had a ROC score at the validation set above 90% even before any hyperparameter tuning. However, this result was outstanding enough to cause skepticism about the validity of the metric used. After a reanalysis of the procedure employed, a data leakage was discovered. Even though the dataset was divided into training and test and a cross-validation method was being employed for hyperparameter tuning, the fact that the same person could participate in the experiment multiple times created an intersection between the set that led to incoherent results. This problem was solved by taking account of the healthcode during the creation of the sets and by avoiding the use of cross-validation with the creation of a validation set.

After applying the solution to the data leakage, the ROC score dropped to about 75% on the validation set even after hyperparameter tuning, while the training ROC score remained above 90% which suggested overfitting. With this 		((((CONTINUE))))

### 3. Unsupervised Learning

[Time Series Classification and Clustering with Python](http://alexminnaar.com/time-series-classification-and-clustering-with-python.html)

[k-Shape](http://www1.cs.columbia.edu/~jopa/Papers/PaparrizosSIGMOD2015.pdf)

### 4. RNN & CNN

[Deep Learning for Time-Series Analysis](https://arxiv.org/abs/1701.01887#)

[Time series classification with Tensorflow](https://burakhimmetoglu.com/2017/08/22/time-series-classification-with-tensorflow/)

### 5. Extra

#### 5.1. Testing features

With the objective of identifying the right number of bins required by some features, a collection of classes ([testFeature](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/testFeature.py)) that allows the visualization of the value of the feature as a function of the number of bins was created. By looking at the plot generated by these classes and recognizing the number from which the value of the functions achieves a plateau, it's possible to spot the right number of bins to be used in the "official" feature generation.

Besides the definition of parameters for the feature calculation, there is another purpose for some of the classes at [testFeature](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/testFeature.py): data analysis from the visualization of the raw data. To fulfill this purpose, there is the class Overview that generates a grid of plots for all the axes of the accelerometer data and also for the magnitude of the resultant acceleration vector.

It's also worth mentioning that the constructor method of the classes mentioned above requires an interaction with the user for the selection of a set of samples from which a random element will be chosen to be loaded. The user can choose between one of the three stages of the experiment (outbound, rest or return) and if it's from a person with or without Parkinson's disease.

#### 5.2. Features visualization

Due to the high dimensionality of the data even after extracting features, two approaches were attempted to reduce the number of dimensions in order to provide a clustering visualization would allow the distinction between time-series from people with and without Parkinson's disease.

The first and simpler method was PCA, to test the viability of this method, it was calculated ([testingPCA](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/Visualization/testingPCA.py)) the variance per axis after fitting the data in a 3-dimensional space. However, the results were very poor (x: 0.18011659, y: 0.10962695, z: 0.08781559) which led to questioning how many dimensions would be enough to achieve a satisfactory variance. Therefore, the data was fitted again but with the variance condition and the result was of 60 dimensions to keep 95% variance, far more than the maximum number for a useful visualization.

To surpass the limitations of a lower dimension projection with PCA, the t-SNE method was applied ([testingTSNE](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/Dimension_Reduction/testingTSNE.py)) and plots for two and three dimensions were generated. However, both figures failed to separate groups data from the two possible targets as can be seen in the pictures below:

| ![2D t-SNE plot](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/Dimension_Reduction/Figures/T-SNE2d.png "2D t-SNE plot") |
|:----:|
| 2D t-SNE plot |


| ![3D t-SNE plot](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Visualization/Dimension_Reduction/Figures/T-SNE3d.png "3D t-SNE plot") |
|:----:|
| 3D t-SNE plot |

### 6. Useful Resources

##### 6.1 Understanding ROC curves and Area Under the Curve

[![ROC Curves and Area Under the Curve (AUC) Explained](https://img.youtube.com/vi/OAl6eAyP-yo/0.jpg)](https://www.youtube.com/watch?v=OAl6eAyP-yo "ROC Curves and Area Under the Curve (AUC) Explained")

##### 6.2 [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

##### 6.3 [Hands-On Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do)

### 7. TODO

- [ ] Completing README
- [ ] Applying smoothing methods
    >Moving Averages Method
- [ ] Testing [tslearn](https://github.com/rtavenar/tslearn)
- [ ] AR models for clustering
    >[Cluster analysis of time series data](http://ai.ms.mff.cuni.cz/~sui/barton.pdf)
----------------