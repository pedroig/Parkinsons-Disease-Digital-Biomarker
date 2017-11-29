import tensorflow as tf
import numpy as np
import rnn_utils as ru
import time
from datetime import datetime

# Log directory for tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

tf.reset_default_graph()

# Hard-coded parameters
timeSeries = ['outbound', 'rest']  # , 'return']
n_steps = 4000
n_inputs = 6  # Rotation Rate (XYZ) and Acceleration (XYZ)
n_neurons = 20
n_outputs = 2
n_layers = 1
learning_rate = 0.001

wavelet = ''  # Empty string for no wavelet
level = 4

dataFractionTrain = 1
dataFractionVal = 1
validateOnOldAgeGroup = True
useDemographics = False

# Training parameters
n_epochs = 30
batch_size = 1000


# Placeholder Tensors
y = tf.placeholder(tf.int32, [None], name="y")
gender = tf.placeholder(tf.float32, [None, 1], name="gender")
age = tf.placeholder(tf.float32, [None, 1], name="age")
X = {}
seq_length = {}
for timeSeriesName in timeSeries:
    with tf.name_scope(timeSeriesName + "_placeholders") as scope:
        X[timeSeriesName] = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        seq_length[timeSeriesName] = tf.placeholder(tf.int32, [None])

# Model
outputs = {}
states = {}
top_layer_h_state = {}
lstm_cells = {}
multi_cell = {}
finalRNNlayers = []
for timeSeriesName in timeSeries:
    with tf.variable_scope(timeSeriesName) as scope:
        lstm_cells[timeSeriesName] = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
                                      for layer in range(n_layers)]
        multi_cell[timeSeriesName] = tf.contrib.rnn.MultiRNNCell(lstm_cells[timeSeriesName])
        outputs[timeSeriesName], states[timeSeriesName] = tf.nn.dynamic_rnn(
            multi_cell[timeSeriesName], X[timeSeriesName], dtype=tf.float32,
            sequence_length=seq_length[timeSeriesName])
        top_layer_h_state[timeSeriesName] = states[timeSeriesName][-1][1]
        finalRNNlayers.append(top_layer_h_state[timeSeriesName])

concat3_top_layer_h_states = tf.concat(finalRNNlayers, axis=1, name="3Stages_concat")
if useDemographics:
    finalLayerInput = tf.concat([concat3_top_layer_h_states, age, gender], axis=1, name="finalLayerInput")
else:
    finalLayerInput = tf.concat([concat3_top_layer_h_states], axis=1, name="finalLayerInput")
logits = tf.layers.dense(finalLayerInput, n_outputs, name="logits")

with tf.name_scope("Cost_function") as scope:
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("Train") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
featuresTableTrain = ru.readPreprocessTable('train')
featuresTableVal = ru.readPreprocessTable('val')

if validateOnOldAgeGroup:
    featuresTableVal = featuresTableVal[featuresTableVal.age > 56]

# Setting size of dataset
featuresTableTrain = featuresTableTrain.sample(frac=dataFractionTrain)
featuresTableVal = featuresTableVal.sample(frac=dataFractionVal)
featuresTableTrain.reset_index(inplace=True)
featuresTableVal.reset_index(inplace=True)

# Reading time series for validation set
X_val, y_val, seq_length_val = ru.generateSetFromTable(featuresTableVal, n_steps, n_inputs, wavelet, level)
feed_dict_val = {
    y: y_val,
    age: np.asarray(featuresTableVal["age"]).reshape((-1, 1)),
    gender: np.asarray(featuresTableVal["Male"]).reshape((-1, 1))
}
for timeSeriesName in timeSeries:
    feed_dict_val[X[timeSeriesName]] = X_val[timeSeriesName]
    feed_dict_val[seq_length[timeSeriesName]] = seq_length_val[timeSeriesName]

n_batches = len(featuresTableTrain) // batch_size

with tf.Session() as sess:

    init.run()

    for epoch in range(n_epochs):

        # reset the local variables used for metrics
        sess.run(tf.local_variables_initializer())

        epoch_start_time = time.time()

        for batch_index in range(n_batches):

            # Building Batch
            featuresTableBatch = featuresTableTrain[featuresTableTrain.index // batch_size == batch_index]
            X_batch, y_batch, seq_length_batch = ru.generateSetFromTable(featuresTableBatch, n_steps, n_inputs, wavelet, level)
            feed_dict_batch = {
                y: y_batch,
                age: np.asarray(featuresTableBatch["age"]).reshape((-1, 1)),
                gender: np.asarray(featuresTableBatch["Male"]).reshape((-1, 1))
            }
            for timeSeriesName in timeSeries:
                feed_dict_batch[X[timeSeriesName]] = X_batch[timeSeriesName]
                feed_dict_batch[seq_length[timeSeriesName]] = seq_length_batch[timeSeriesName]

            # Training operation and metrics updates
            sess.run([training_op, auc_update_op, precision_update_op, recall_update_op], feed_dict=feed_dict_batch)

            # Tensorboard summary
            if batch_index % 4 == 0:
                summary_str = loss_summary.eval(feed_dict=feed_dict_batch)
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)

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

    save_path = saver.save(sess, "./checkpoints/run-{}/model.ckpt".format(now))

file_writer.close()
