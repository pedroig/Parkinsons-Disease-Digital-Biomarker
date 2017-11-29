import tensorflow as tf
import cnn_utils as cnnu

# Hard-coded parameters
channels_input = 3
n_outputs = 2
timeSeriesPaddedLength = 4000
timeSeries = 'rest'
wavelet = ''
level = 4
validateOnOldAgeGroup = True
folderCode = '20171129030924'

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
    init = tf.group(tf.local_variables_initializer())
    saver = tf.train.Saver()

featuresTableVal = cnnu.readPreprocessTable('test')

if validateOnOldAgeGroup:
    featuresTableVal = featuresTableVal[featuresTableVal.age > 56]

# Reading time series for validation set
X_val, y_val = cnnu.generateSetFromTable(featuresTableVal, wavelet, level, timeSeries, channels_input, timeSeriesPaddedLength)
feed_dict_val = {
    y: y_val,
    X: X_val
}

with tf.Session() as sess:

    # initialize all the trainable variables and all the local variables used for metrics
    init.run()
    saver.restore(sess, "./checkpoints/run-{}/model.ckpt".format(folderCode))

    print("\tValidation")
    sess.run(tf.local_variables_initializer())
    precision_val, auc_val, recall_val = sess.run([precision_update_op, auc_update_op, recall_update_op], feed_dict=feed_dict_val)
    print("\t\tROC AUC:", auc_val)
    print("\t\tPrecision:", precision_val[0])
    print("\t\tRecall:", recall_val[0])
