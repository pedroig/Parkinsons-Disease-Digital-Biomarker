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

* The second type of data augmentation was the use of a random 3D rotation to simulate people holding the phone in different orientations which would correspond to different world frame coordinate systems. However, it is important to highlight that the rotation is limited to the plane orthogonal to the z-axis since this axis is always well defined by the gravity in the world's frame. It is also worth mentioning that this rotation is not contradictory with the rotation executed to convert the data from the phone's frame to the world's frame. This is explained by the fact that the conversion rotation was based in the quaternions from the gyroscope data which changes for each timestamp while the augmentation uses a single quaternion, applying the same corresponding rotation to all the samples in a 3D time series.

The code that performs the data augmentation described above can be found in the [function augmentRow](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/utils.py) that generates JSON augmented versions for the rotation rate samples of one sequence of measurement, this function is then parallelized by the [apply_by_multiprocessing_augmentation function](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/utils.py) in order to be applied to all the data samples. An augmented version of the training table is created by the [function generateAugmentedTable](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/utils.py) that receives as parameter the fraction of the training data that is going to have the augmented version used.

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

#### 1.6 Splitting the Dataset

After executing the initial cleaning procedures and generating the features described in 1.4, a new version of the original walking_activity.csv table is saved as walking_activity_features.csv, allowing the dataset to be easily split without generating the features everytime. The final cleaning routine and the split procedure is done by the [generateSetTables function](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/splitSets.py) that divides the dataset in the following configuration: 80% Training set, 10% Validation set, and 10% Test set. However, it is critical to call attention to the fact that the split is done by taking account of the healthcode during the creation of the sets, i.e., splitting the demographics table and then merging it to the walking_activity table. This approach is necessary in order to avoid data leakage since otherwise, the same person could be at the same time in the training set and in the validation set which would lead to unrealistic good results. It is also important to highlight that taking account of the healthCode means that the final sizes of the sets are not exactly coherent with the previously stipulated percentages since different people performed the experiment different times.

One undesirable property observed in the final sets is the unbalanced distribution between labels. From a total of 27,314 samples after cleaning, 17,663 samples correspond to people with Parkinson's disease. This gap gets even wider when analyzing only people older 56 years old: 14,907 people with the disease from a total of 17,933.

#### 1.7 Code Structure

The code responsible for the feature generation is divided into 3 parts:
* [cleanFeaturise](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/cleanFeaturise.py): Accesses the walking activity CSV table, executes the first stages described for data cleaning, performs the feature generation and saves the results in a new table.
* [splitSets](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/splitSets.py): Accesses the demographics table and also the table generated by the cleanFeaturise code, performs the merge operation taking the healthCode in consideration and also executes the final steps of data cleaning.
* [createFeatures](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/createFeatures.py): the Main overview of all the features generated for each sample.
* [features_utils](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/features_utils.py): Collection of functions used to perform the specific operations for each feature.
* [quaternion](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/quaternion.py): Quaternion class with the operations to perform the rotation.
* [utils](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/utils.py): Utility functions used for file manipulation or data preprocessing.
* [run](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Features/run.py): Top-level script to execute the cleanFeaturise code, splitSets code or the data augmentation.

### 2 Machine Learning Modeling

#### 2.1. Random Forest

The first and simpler model developed to tackle this problem was a Random Forest model that uses all the features extracted from the time-series, the gender and the age as input. This model was further enhanced by the use of an ensemble to deal with the unbalanced distribution of labels in the dataset. The ensemble design makes use of the undersampling technique, training each random forest in a different fifty-fifty balanced dataset in which the majority class is randomly selected to match the size of the minority class. The ensemble structure is illustrated below:

| ![Ensemble Diagram](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Extra/Figures/Ensemble_Diagram.png "Ensemble Diagram") |
|:----:|
| Ensemble Diagram |

The code for the random forest model can be found in the [function randomForestModel](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Random_Forest/randomForest.py) which also supports non ensemble options with optional training set balancing with undersampling or [SMOTE](http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html#from-random-over-sampling-to-smote-and-adasyn) oversampling. Besides providing standart metrics results about the performance of the model, the randomForestModel function also gives feedback about the feature importance (the normalized total reduction of the criterion brought by that feature) ranking in which can be observed that the demographic features have much higher priorities in the greedy algorithm that builds the decision trees in each random forest ensemble. Due to the randomization in the model, the feature importance ranking is not constant, but the plot and the table below offer a preview of the importance distribution accross the features:

| ![Feature Importance Ranking](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Extra/Figures/Importances_Ranking.png "Feature Importance Ranking") |
|:----:|
| Feature Importance Ranking |

##### 20 Most Important Features

| Ranking | Name | Stage | Axis (es) | Measurent Type | Importance |
|-------------|-------------|-------------|-------------|-------------|-------------|
1 | Age | - | - | - | 0.254625 |
2 | Gender | - | - | - | 0.084887 |
3 | Dominant Frequency Component | Outbound | Z | Acceleration | 0.034564 |
4 | Dominant Frequency Component | Outbound | X | Rotation Rate | 0.026673 |
5 | Dominant Frequency Component | Outbound | Z | Rotation Rate | 0.022937 |
6 | Dominant Frequency Component | Rest | Z | Rotation Rate | 0.020182 |
7 | Kurtosis | Rest | Z | Rotation Rate | 0.018007 |
8 | Dominant Frequency Component | Outbound | Y | Rotation Rate | 0.016551 |
9 | Maximum | Outbound | Z | Acceleration | 0.011511 |
10 | Interquartile | Rest | Z | Acceleration | 0.011406 |
11 | Mutual Information | Rest | YZ | Acceleration | 0.010912 |
12 | Mutual Information | Outbound | XY | Rotation Rate | 0.010891 |
13 | Mutual Information | Rest | YZ | Rotation Rate | 0.010618 |
14 | Q1 | Rest | Y | Rotation Rate | 0.008552 |
15 | Zero Crossing Rate | Rest | Y | Acceleration | 0.008523 |
16 | Entropy | Rest | Y | Acceleration | 0.008482 |
17 | Q3 | Outbound | Z | Acceleration | 0.008227 |
18 | Data Range | Outbound | Y | Acceleration | 0.007834 |
19 | Data Range | Outbound | Z | Acceleration | 0.007720 |
20 | Entropy | Rest | X | Rotation Rate |  0.007604 |

The table above displays a tendency to obtain higher importance values for features extracted from the z-axis of the time-series. This could be explained by the fact that this is axis is well defined in the world frame coordinate system as mentioned when discussing the data augmentation.

In order to allow an easier way to tune the hyperparameters from the random forest, it was developed an auxiliary [randomForestTuning function](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Random_Forest/randomForest.py) that plots the model performance for a range of hyperparameters values. While one hyperparameter chosen by the user is plotted in a range of values, the other ones are fixed to standard quantities. Three hyperparameters can be analyzed in this procedure:
* The maximum depth of the tree;
* The number of trees in the forest;
* The minimum number of samples required to split an internal node.

The graph plotted below elucidates the tuning visualization offered:

| ![Maximum Depth Tuning](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Random_Forest/Forest_Graphs/roc_score_max_depth_gini_undersampling_ValAbove56years.png "Maximum Depth Tuning") |
|:----:|
| Maximum Depth Tuning |

#### 2.2 Convolutional Neural Network

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

It's also worth mentioning that the constructor method of the classes mentioned above requires an interaction with the user for the selection of a set of samples from which a random element will be chosen to be loaded. The user has the following options:
* Choose between one of the three stages of the experiment (outbound, rest or return);
* If the data is from a person with or without Parkinson's disease;
* Choose between time-series data of the rotation rate or the acceleration.

The picture below illustrates the main functionalities of the Overview class using the rotation rate data from a Parkinson's disease patient in the walking outbound stage:

| ![Overview example](https://github.com/pedroig/Parkinsons-Disease-Digital-Biomarker/blob/master/Extra/Figures/VisualizationExample.png "Overview example") |
|:----:|
| Overview example |

In the example above the vertical bars create segments in the time-series corresponding to each step taken by the volunteer and a smoothed version of the time-series using Daubechies 9 at level 4 was plotted for a comparison with the raw variant.

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

##### 3.3.1 [Getting Started on NOTS](https://docs.rice.edu/confluence/display/CD/Getting+Started+on+NOTS)

##### 3.3.2 [Hands-On Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do)

##### 3.3.3 [PyWavelets Manual](https://media.readthedocs.org/pdf/pywavelets/latest/pywavelets.pdf)

##### 3.3.4 [Understanding ROC curves and Area Under the Curve](https://www.youtube.com/watch?v=OAl6eAyP-yo)

### 4 Instructions to run the code

If you have access to the Parkinson's Disease Digital Biomarker dataset, the results obtained in this project can be reproduced by the following sequence of steps:

* Clone or download the repository.

* Place the demographics.csv and walking_activity.csv tables in the directory ./data/  .

* Place the JSON files in a directory with the following pattern: ./data/{timeSeriesName}/{last three digits of the pointer in the walking_actibity table}/{pointer number}/ . The parameter timeSeriesName has three possible values: 'deviceMotion_walking_outbound', 'deviceMotion_walking_rest' or 'pedometer_walking_outbound' .

* Run the /Features/run.py code three times with the following sequence of command-line arguments:
	1. 'cleanFeaturise'
	2. 'splitSets' (Note: possible warnings should be ignored)
	3. 'augmentation'

	It's important to take in mind that to execute it with the parameters _i_ and _iii_ without parallelization could take days. On the other hand, the execution with parameter 2 takes about two minutes, allowing a quick reuse afterward if the user intends to generate a new dataset distribution for the Training, Validation and Test sets.

Now, all the data is already set up and the user can safely run the Random Forest Model or the Convolutional Model.

----------------

>This work was supported in part by the Big-Data Private-Cloud Research Cyberinfrastructure MRI-award funded by NSF under grant CNS-1338099 and by Rice University.