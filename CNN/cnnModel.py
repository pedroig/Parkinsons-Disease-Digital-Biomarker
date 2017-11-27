import tensorflow as tf
import cnn_utils as cnnu
import numpy as np
from datetime import datetime
from sklearn import metrics

# Log directory for tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

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
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    positiveClass_probability = tf.sigmoid(logits[:, 1] - logits[:, 0])

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()  # prepare an init node
    saver = tf.train.Saver()


# Summary definition for tensorboard
loss_summary = tf.summary.scalar('Loss', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Reading tables
featuresTableTrain = cnnu.readPreprocessTable('train')
featuresTableVal = cnnu.readPreprocessTable('val')

if validateOnOldAgeGroup:
    featuresTableVal = featuresTableVal[featuresTableVal.age > 56]

# Setting size of dataset
featuresTableTrain = featuresTableTrain.sample(frac=dataFractionTrain)
featuresTableVal = featuresTableVal.sample(frac=dataFractionVal)

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
max_checks_without_progress = 10

with tf.Session() as sess:
    init.run()  # actually initialize all the variables
    for epoch in range(n_epochs):
        acc_train = np.array([])
        auc_train = np.array([])
        for batch_index in range(n_batches):

            # Building Batch
            featuresTableBatch = featuresTableTrain[featuresTableTrain.index // batch_size == batch_index]
            X_batch, y_batch = cnnu.generateSetFromTable(featuresTableVal, wavelet, level, timeSeries, channels_input, timeSeriesPaddedLength)
            feed_dict_batch = {
                y: y_batch,
                X: X_batch
            }

            # Training operation
            _, acc_batch, prob_batch = sess.run([training_op, accuracy, positiveClass_probability], feed_dict=feed_dict_batch)
            acc_train = np.append(acc_train, acc_batch)
            auc_train = np.append(auc_train, metrics.roc_auc_score(y_batch, prob_batch))

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

        # Validation set metrics for current epoch
        acc_val, y_prob = sess.run([accuracy, positiveClass_probability], feed_dict=feed_dict_val)
        auc_val = metrics.roc_auc_score(y_val, y_prob)
        print("Epoch:", epoch)
        print("\tTraining")
        print("\t\tAccuracy:", acc_train.mean())
        print("\t\tROC AUC:", auc_train.mean())
        print("\tValidation")
        print("\t\tAccuracy:", acc_val)
        print("\t\tROC AUC:", auc_val)

        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

file_writer.close()
