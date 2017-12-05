import tensorflow as tf
import cnn_utils as cnnu
import numpy as np
import time
from datetime import datetime

# Hard-coded parameters
batch_size = 4
learning_rate = 0.0005
n_epochs = 50
channels_input = 3
n_outputs = 2
timeSeriesPaddedLength = 4000
timeSeries = 'rest'
wavelet = ''
level = 4
dataFractionTrain = 1
dataFractionVal = 1
validateOnOldAgeGroup = True

# Log directory for tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
folderName = "run-{}_{}_epochs-{}_learningRate-{}_batchSize-{}".format(now, timeSeries, n_epochs, learning_rate, batch_size)
if wavelet is not '':
    folderName += "_{}{}".format(wavelet, level)
logdir = "{}/{}/".format(root_logdir, folderName)

tf.reset_default_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, timeSeriesPaddedLength, channels_input], name="X")
    y = tf.placeholder(tf.int32, shape=[None], name="y")

convFilters = [8, 16, 32, 32, 64, 64, 128, 128]
convKernelSizes = [5, 5, 4, 4, 4, 4, 4, 5]
conv = {}
pool = {}

for layerNumber in range(8):
    if layerNumber == 0:
        conv[layerNumber] = tf.layers.conv1d(inputs=X, filters=convFilters[layerNumber],
                                             kernel_size=convKernelSizes[layerNumber],
                                             strides=1, padding='valid',
                                             data_format='channels_last',
                                             activation=tf.nn.relu, name="conv1")
    else:
        conv[layerNumber] = tf.layers.conv1d(inputs=pool[layerNumber - 1],
                                             filters=convFilters[layerNumber],
                                             kernel_size=convKernelSizes[layerNumber],
                                             strides=1, padding='valid',
                                             data_format='channels_last',
                                             activation=tf.nn.relu,
                                             name="conv{}".format(layerNumber + 1))

    pool[layerNumber] = tf.layers.max_pooling1d(inputs=conv[layerNumber], pool_size=2,
                                                strides=2, padding='valid',
                                                data_format='channels_last',
                                                name='pool{}'.format(layerNumber + 1))

with tf.name_scope("output"):
    flat = tf.reshape(pool[7], shape=[-1, 12 * 128])
    logits = tf.layers.dense(inputs=flat, units=n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    positiveClass_probability = tf.sigmoid(logits[:, 1] - logits[:, 0])
    auc, auc_update_op = tf.metrics.auc(labels=y, predictions=positiveClass_probability, num_thresholds=10000)
    precision, precision_update_op = tf.metrics.precision_at_thresholds(labels=y, thresholds=[0.5],
                                                                        predictions=positiveClass_probability)
    recall, recall_update_op = tf.metrics.recall_at_thresholds(labels=y, thresholds=[0.5],
                                                               predictions=positiveClass_probability)

with tf.name_scope("init_and_save"):
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()

with tf.name_scope("tensorboard"):
    loss_summary = tf.summary.scalar('Loss', loss)
    auc_train_summary = tf.summary.scalar('AUC_Training', auc)
    auc_val_summary = tf.summary.scalar('AUC_Validation', auc)
    precision_train_summary = tf.summary.scalar('Precision_Training', precision[0])
    precision_val_summary = tf.summary.scalar('Precision_Validation', precision[0])
    recall_train_summary = tf.summary.scalar('Recall_Training', recall[0])
    recall_val_summary = tf.summary.scalar('Recall_Validation', recall[0])
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Reading tables
featuresTableTrain = cnnu.readPreprocessTable('train')
featuresTableVal = cnnu.readPreprocessTable('val')

if validateOnOldAgeGroup:
    featuresTableVal = featuresTableVal[featuresTableVal.age > 56]

# Setting size of dataset
featuresTableTrain = featuresTableTrain.sample(frac=dataFractionTrain)
featuresTableVal = featuresTableVal.sample(frac=dataFractionVal)
featuresTableTrain.reset_index(inplace=True)
featuresTableVal.reset_index(inplace=True)

# Reading time series for validation set
X_val, y_val = cnnu.generateSetFromTable(featuresTableVal, wavelet, level, timeSeries, channels_input, timeSeriesPaddedLength)
feed_dict_val = {
    y: y_val,
    X: X_val
}

n_batches = len(featuresTableTrain) // batch_size
best_loss_val = np.infty
check_interval = 50
checks_since_last_progress = 0
max_checks_without_progress = 500

with tf.Session() as sess:

    # initialize all the trainable variables and all the local variables used for metrics
    init.run()

    for epoch in range(n_epochs):

        # reset the local variables used for metrics
        sess.run(tf.local_variables_initializer())

        epoch_start_time = time.time()

        for batch_index in range(n_batches):

            # Building Batch
            featuresTableBatch = featuresTableTrain[featuresTableTrain.index // batch_size == batch_index]
            X_batch, y_batch = cnnu.generateSetFromTable(featuresTableBatch, wavelet, level, timeSeries, channels_input, timeSeriesPaddedLength)
            feed_dict_batch = {
                y: y_batch,
                X: X_batch
            }

            # Training operation and metrics updates
            sess.run([training_op, auc_update_op, precision_update_op, recall_update_op], feed_dict=feed_dict_batch)

            # Tensorboard summary
            if batch_index % 4 == 0:
                summary_str = loss_summary.eval(feed_dict=feed_dict_batch)
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)

            if batch_index % check_interval == 0:
                loss_val = loss.eval(feed_dict=feed_dict_val)
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                else:
                    checks_since_last_progress += 1

        # Metrics
        print("Epoch: {}, Execution time: {} seconds".format(epoch, time.time() - epoch_start_time))

        # Metrics on training data
        print("\tTraining")
        file_writer.add_summary(auc_train_summary.eval(), epoch)
        file_writer.add_summary(precision_train_summary.eval(), epoch)
        file_writer.add_summary(recall_train_summary.eval(), epoch)
        print("\t\tROC AUC:", auc.eval())
        print("\t\tPrecision:", precision.eval()[0])
        print("\t\tRecall:", recall.eval()[0])

        # Validation set metrics for current epoch
        print("\tValidation")
        sess.run(tf.local_variables_initializer())
        precision_val, auc_val, recall_val = sess.run([precision_update_op, auc_update_op, recall_update_op], feed_dict=feed_dict_val)
        file_writer.add_summary(auc_val_summary.eval(), epoch)
        file_writer.add_summary(precision_val_summary.eval(), epoch)
        file_writer.add_summary(recall_val_summary.eval(), epoch)
        print("\t\tROC AUC:", auc_val)
        print("\t\tPrecision:", precision_val[0])
        print("\t\tRecall:", recall_val[0])

        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    save_path = saver.save(sess, "./checkpoints/{}/model.ckpt".format(folderName))

file_writer.close()
