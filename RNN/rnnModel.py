import tensorflow as tf
import numpy as np
import rnn_utils as ru
from datetime import datetime
from sklearn import metrics

# Log directory for tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

tf.reset_default_graph()

# Choosing time-series to read
wavelet = 'db9'  # Empty string for no wavelet
level = 4

# Hard-coded parameters
timeSeries = ['outbound', 'rest', 'return']
n_steps = {
    'outbound': 3159,
    'rest': 3603,
    'return': 3226
}
n_inputs = 6  # Rotation Rate (XYZ) and Acceleration (XYZ)
n_neurons = 20
n_outputs = 2
n_layers = 1
learning_rate = 0.001


# Placeholder Tensors
y = tf.placeholder(tf.int32, [None], name="y")
gender = tf.placeholder(tf.float32, [None, 1], name="gender")
age = tf.placeholder(tf.float32, [None, 1], name="age")
X = {}
seq_length = {}
for timeSeriesName in timeSeries:
    with tf.name_scope(timeSeriesName + "_placeholders") as scope:
        X[timeSeriesName] = tf.placeholder(tf.float32, [None, n_steps[timeSeriesName], n_inputs])
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
finalLayerInput = tf.concat([concat3_top_layer_h_states, age, gender], axis=1, name="finalLayerInput")
logits = tf.layers.dense(finalLayerInput, n_outputs, name="logits")

with tf.name_scope("Cost_function") as scope:
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("Train") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("Metrics") as scope:
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    prediction = tf.argmax(logits, axis=1)

init = tf.global_variables_initializer()  # prepare an init node


# Summary definition for tensorboard
loss_summary = tf.summary.scalar('Loss', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Reading tables
featuresTableTrain = ru.readPreprocessTable('train')
featuresTableVal = ru.readPreprocessTable('val')

# Setting size of dataset
n_train_size = 15000
n_val_size = 1773
featuresTableTrain = featuresTableTrain.iloc[:n_train_size]
featuresTableVal = featuresTableVal.iloc[:n_val_size]

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

# Training parameters
n_epochs = 12
batch_size = 1000
n_batches = len(featuresTableTrain) // batch_size

with tf.Session() as sess:
    init.run()  # actually initialize all the variables
    for epoch in range(n_epochs):
        acc_train = np.array([])
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

            # Training operation
            _, acc_batch = sess.run([training_op, accuracy], feed_dict=feed_dict_batch)
            acc_train = np.append(acc_train, acc_batch)

            # Tensorboard summary
            if batch_index % 4 == 0:
                summary_str = loss_summary.eval(feed_dict=feed_dict_batch)
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)

        # Validation set metrics for current epoch
        acc_val, y_pred = sess.run([accuracy, prediction], feed_dict=feed_dict_val)
        auc_val = metrics.roc_auc_score(y_val, y_pred)
        print(epoch, "Train accuracy:", acc_train.mean(), "Validation accuracy:", acc_val, "Validation AUC score:", auc_val)

file_writer.close()
