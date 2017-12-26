import functools
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import sys
from datetime import datetime
sys.path.insert(0, '../Features')
import utils


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class CNN:

    def __init__(self,
                 learning_rate=0.0001,
                 batch_size=100,
                 n_epochs=50,
                 timeSeries='rest',
                 wavelet='',
                 level=4,
                 dataFractionTrain=1,
                 dataFractionVal=1,
                 validateOnOldAgeGroup=True,
                 check_interval=50,
                 max_checks_without_progress=500,
                 developmentSet='val',
                 restoreFolderName='',
                 useAugmentedData=False):
        """
        Input:
            - learning_rate: float
                real positive number
            - batch_size: int
                integer >= 1
            - n_epochs: int
                integer >=1
            - timeSeries: string
                'rest' or 'outbound'
            - wavelet: string
                Wavelet to use, empty string if no wavelet is used for smoothing.
                example: 'db9'
            - level: integer
                Decomposition level for the wavelet. This parameter is not considered if no wavelet is used.
            - dataFractionTrain: float
                0 < dataFractionTrain <= 1
            - dataFractionVal: float
                0 < dataFractionVal <= 1
            - validateOnOldAgeGroup: bool
                Whether to select only people older 56 years in the validation set.
            - check_interval: integer
            - max_checks_without_progress: integer
            - developmentSet: string
                'val' or 'test'
            - restoreFolderName: string
                Folder name for checkpoint to be restored. If the string is empty, nothing is restored
            - useAugmentedData: bool
                Whether to use augmented data in the training set.
        """
        self.channels_input = 3
        self.n_outputs = 2
        self.timeSeriesPaddedLength = 4000
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.timeSeries = timeSeries
        self.wavelet = wavelet
        self.level = level
        self.restoreFolderName = restoreFolderName
        self.developmentSet = developmentSet
        self.useAugmentedData = useAugmentedData

        self.generateDirectoriesNames()

        self.logits_prediction
        self.optimize
        self.metrics
        self.tensorboard_summaries
        self.init_and_save

        self.loadTables(validateOnOldAgeGroup, dataFractionTrain, dataFractionVal)

        # Reading time series for validation set
        X_val, y_val = self.generateSetFromTable(self.featuresTableVal)
        self.feed_dict_val = {
            self.y: y_val,
            self.X: X_val
        }

        self.n_batches = len(self.featuresTableTrain) // self.batch_size
        self.best_loss_val = np.infty
        self.checks_since_last_progress = 0
        self.check_interval = check_interval
        self.max_checks_without_progress = max_checks_without_progress

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def logits_prediction(self):
        """
        Outputs logits tensor, i.e., the unscaled log probabilities for each of the two possible classes.
        """
        self.X = tf.placeholder(tf.float32, shape=[None, self.timeSeriesPaddedLength, self.channels_input], name="X")
        self.y = tf.placeholder(tf.int32, shape=[None], name="label")

        convFilters = [8, 16, 32, 32, 64, 64, 128, 128]
        convKernelSizes = [5, 5, 4, 4, 4, 4, 4, 5]

        x = self.X
        for layerNumber in range(8):
            x = tf.layers.conv1d(inputs=x,
                                 filters=convFilters[layerNumber],
                                 kernel_size=convKernelSizes[layerNumber],
                                 strides=1, padding='valid',
                                 data_format='channels_last',
                                 activation=tf.nn.relu,
                                 name="conv{}".format(layerNumber + 1))

            x = tf.layers.max_pooling1d(inputs=x, pool_size=2,
                                        strides=2, padding='valid',
                                        data_format='channels_last',
                                        name='pool{}'.format(layerNumber + 1))

        flat = tf.reshape(x, shape=[-1, 12 * 128], name="flat")
        logits = tf.layers.dense(inputs=flat, units=self.n_outputs, name="logits")
        return logits

    @define_scope("Train")
    def optimize(self):
        """
        Outputs the gradient descent operation.
        """
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_prediction, labels=self.y)
        self.loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.loss)

    @define_scope("Metrics")
    def metrics(self):
        """
        Outputs grouped update operation for the following metrics:
            * Area under the ROC curve;
            * Precision;
            * Recall.

        It's important to highlight that the update operation accumulates the relevant data from the confusion matrix for each metric
        in local variables, enabling the estimation of the selected metrics over a stream of data like the mini-batches. For this
        reason, it is required to reset the local variables before calling this method if the intention is to make an estimation
        in a different dataset.
        """
        logits = self.logits_prediction
        positiveClass_probability = tf.sigmoid(logits[:, 1] - logits[:, 0])
        self.auc, auc_update_op = tf.metrics.auc(labels=self.y, predictions=positiveClass_probability, num_thresholds=10000)
        self.precision, precision_update_op = tf.metrics.precision_at_thresholds(labels=self.y, thresholds=[0.5],
                                                                                 predictions=positiveClass_probability)
        self.recall, recall_update_op = tf.metrics.recall_at_thresholds(labels=self.y, thresholds=[0.5],
                                                                        predictions=positiveClass_probability)
        update_ops = tf.group(auc_update_op, precision_update_op, recall_update_op)
        return update_ops

    @define_scope("Tensorboard")
    def tensorboard_summaries(self):
        """
        Tensor summaries for exporting information about the model to tensorboard.
        """
        self.loss_summary = tf.summary.scalar('Loss', self.loss)
        self.auc_summary = {
            'Training': tf.summary.scalar('AUC_Training', self.auc),
            'Validation': tf.summary.scalar('AUC_Validation', self.auc)
        }
        self.precision_summary = {
            'Training': tf.summary.scalar('Precision_Training', self.precision[0]),
            'Validation': tf.summary.scalar('Precision_Validation', self.precision[0])
        }
        self.recall_summary = {
            'Training': tf.summary.scalar('Recall_Training', self.recall[0]),
            'Validation': tf.summary.scalar('Recall_Validation', self.recall[0])
        }
        self.file_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())

    @define_scope("init_and_save")
    def init_and_save(self):
        """
        Auxiliar tensorflow nodes:
            * Node in the graph that initializes all variables when it is run;
            * Saver node to save and restore variables to and from checkpoints.
        """
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def summary_per_minibatch(self, batch_index, epoch, feed_dict_batch):
        if batch_index % 4 == 0:
            summary_str = self.loss_summary.eval(feed_dict=feed_dict_batch)
            step = epoch * self.n_batches + batch_index
            self.file_writer.add_summary(summary_str, step)

    def process_summaries_set(self, setName, epoch):
        """
        Saves the metrics from the current epoch in the tensorboard summaries and prints it for the user.

        Input:
            - setName: string
                String to select which summaries to process: 'Training' or 'Validation'.
            - epoch: int
                Epoch number that corresponds to the horizontal axis when plotting the summaries.
        """
        self.file_writer.add_summary(self.auc_summary[setName].eval(), epoch)
        self.file_writer.add_summary(self.precision_summary[setName].eval(), epoch)
        self.file_writer.add_summary(self.recall_summary[setName].eval(), epoch)
        self.printMetrics(setName)

    def printMetrics(self, setName):
        """
        Input:
            - setName: string
                String to be printed to inform the user which set the metrics are from: 'train', val' or 'test'.
        """
        print("\t{}".format(setName))
        print("\t\tROC AUC:", self.auc.eval())
        print("\t\tPrecision:", self.precision.eval()[0])
        print("\t\tRecall:", self.recall.eval()[0])

    def evaluateMetricsRestored(self):
        """
        Restores trainable parameters correspondent to the folder name specified in the constructor method and evaluates
        the performance of the model in the development set specified (validation or test).
        """
        with tf.Session() as sess:
            # reset the local variables used for metrics
            sess.run(tf.local_variables_initializer())
            if len(self.restoreFolderName) > 0:
                self.saver.restore(sess, "./checkpoints/{}/model.ckpt".format(self.restoreFolderName))
                sess.run(self.metrics, feed_dict=self.feed_dict_val)
                self.printMetrics(self.developmentSet)

    def train(self):
        """
        Executes the tensorflow graph to train the model while also saving and displaying metrics of the process.

        It is important to highlight that the mini-batches are loaded to memory on demand, making it so that only
        one is in memory at any given time.
        """
        with tf.Session() as sess:

            if len(self.restoreFolderName) > 0:
                self.saver.restore(sess, "./checkpoints/{}/model.ckpt".format(self.restoreFolderName))
            else:
                self.init.run()

            for epoch in range(self.n_epochs):

                # reset the local variables used for metrics
                sess.run(tf.local_variables_initializer())

                epoch_start_time = time.time()

                for batch_index in range(self.n_batches):

                    # Building Batch
                    featuresTableBatch = self.featuresTableTrain[self.featuresTableTrain.index // self.batch_size == batch_index]
                    X_batch, y_batch = self.generateSetFromTable(featuresTableBatch)
                    feed_dict_batch = {
                        self.y: y_batch,
                        self.X: X_batch
                    }

                    # Training operation and metrics updates
                    sess.run([self.optimize, self.metrics], feed_dict=feed_dict_batch)

                    self.summary_per_minibatch(batch_index, epoch, feed_dict_batch)

                    if batch_index % self.check_interval == 0:
                        loss_val = self.loss.eval(feed_dict=self.feed_dict_val)
                        if loss_val < self.best_loss_val:
                            self.best_loss_val = loss_val
                            checks_since_last_progress = 0
                        else:
                            checks_since_last_progress += 1

                # Metrics
                print("Epoch: {}, Execution time: {} seconds".format(epoch, time.time() - epoch_start_time))

                # Metrics on training data
                self.process_summaries_set("Training", epoch)

                # Validation set metrics for current epoch
                sess.run(tf.local_variables_initializer())
                sess.run(self.metrics, feed_dict=self.feed_dict_val)
                self.process_summaries_set("Validation", epoch)

                if self.checks_since_last_progress > self.max_checks_without_progress:
                    print("Early stopping!")
                    break

            self.saver.save(sess, self.checkpointdir)

        self.file_writer.close()

    def generateDirectoriesNames(self):
        """
        Generates names for:
            *Checkpoint directory;
            *Log directory for tensorboard.
        """
        self.now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        folderName = "run-{}_{}_epochs-{}_learningRate-{}_batchSize-{}".format(
            self.now,
            self.timeSeries,
            self.n_epochs,
            self.learning_rate,
            self.batch_size)
        if self.wavelet is not '':
            folderName += "_{}{}".format(self.wavelet, self.level)
        if self.useAugmentedData:
            folderName += "_augmented"
        self.logdir = "tf_logs/{}/".format(folderName)
        self.checkpointdir = "./checkpoints/{}/model.ckpt".format(folderName)

    def readPreprocessTable(self, name):
        """
        Input:
            - name: string
                Table to be loaded: 'train', val' or 'test'.
        """
        featuresTable = pd.read_csv("../data/{}_extra_columns.csv".format(name), index_col=0)
        # Renaming to use the column name to access a named tuple
        for timeSeriesName in ['outbound', 'rest']:  # , 'return']:
            featuresTable.rename(columns={'deviceMotion_walking_{}.json.items'.format(timeSeriesName):
                                          'deviceMotion_walking_' + timeSeriesName},
                                 inplace=True)
        featuresTable.reset_index(inplace=True)
        return featuresTable

    def loadTables(self, validateOnOldAgeGroup, dataFractionTrain, dataFractionVal):
        """
        Loads CSV the tables with the pointers to JSON files, optionally filtering the development set table to only
        have people in the group of main interest for prediction, older people, here specified to be above 56 years old.
        Fraction parameters are given in case there is a need to limit the amount of data to be loaded for a quick
        testing/debugging of the code.
        """

        # Reading tables
        if self.useAugmentedData:
            self.featuresTableTrain = self.readPreprocessTable('train_augmented')
        else:
            self.featuresTableTrain = self.readPreprocessTable('train')
        self.featuresTableVal = self.readPreprocessTable(self.developmentSet)

        if validateOnOldAgeGroup:
            self.featuresTableVal = self.featuresTableVal[self.featuresTableVal.age > 56]

        # Setting size of the dataset
        self.featuresTableTrain = self.featuresTableTrain.sample(frac=dataFractionTrain)
        self.featuresTableVal = self.featuresTableVal.sample(frac=dataFractionVal)

        self.featuresTableTrain.reset_index(inplace=True)
        self.featuresTableVal.reset_index(inplace=True)

    def generateSetFromTable(self, featuresTable):
        """
        Loads all the rotation rate JSON files from a given table into memory.
        """
        fileNameRotRate = utils.genFileName('RotRate', self.wavelet, self.level)
        axes = ['x', 'y', 'z']
        y = featuresTable.Target
        y = np.array(y)
        X = {}
        timeSeriesName = 'deviceMotion_walking_' + self.timeSeries
        X = pd.DataFrame(columns=axes)
        for row in featuresTable.itertuples():
            if "augmented" in featuresTable and row.augmented:
                data = utils.readJSON_data(getattr(row, timeSeriesName), timeSeriesName, "RotRate_augmented.json")
            else:
                data = utils.readJSON_data(getattr(row, timeSeriesName), timeSeriesName, fileNameRotRate)
            XElement = data.loc[:, axes]
            zeros = pd.DataFrame(0, index=np.arange(self.timeSeriesPaddedLength - len(data)), columns=axes)
            X = pd.concat([X, XElement, zeros])
        X = np.asarray(X)
        X = X.reshape((-1, self.timeSeriesPaddedLength, self.channels_input))
        return X, y


def main():

    tf.reset_default_graph()

    # model = CNN(restoreFolderName='run-20171217013755_rest_epochs-50_learningRate-0.0001_batchSize-100', developmentSet='test')
    # model.evaluateMetricsRestored()

    model = CNN(learning_rate=0.0001,
                batch_size=100,
                timeSeries='rest',
                useAugmentedData=False)
    model.train()


if __name__ == '__main__':
    main()
