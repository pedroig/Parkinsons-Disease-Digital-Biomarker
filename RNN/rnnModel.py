import tensorflow as tf
import numpy as np
import rnn_utils as ru

tf.reset_default_graph()

n_steps = 3603  # maximum time-series length for the rest stage
n_inputs = 3
n_neurons = 20
n_outputs = 2
n_layers = 2

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
seq_length = tf.placeholder(tf.int32, [None])


lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
              for layer in range(n_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=seq_length)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()


featuresTableTrain = ru.readPreprocessTable('train')
featuresTableVal = ru.readPreprocessTable('val')

n_train_size = 1000  # max = 15001
n_val_size = 1700  # max = 1773

featuresTableTrain = featuresTableTrain.iloc[:n_train_size]
featuresTableVal = featuresTableVal.iloc[:n_val_size]

X_val, y_val, seq_length_val = ru.generateSetFromTable(featuresTableVal, n_steps, n_inputs)


n_epochs = 10
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        acc_train = np.array([])
        for iteration in range(len(featuresTableTrain) // batch_size):
            featuresTableBatch = featuresTableTrain[featuresTableTrain.index // batch_size == iteration]
            X_batch, y_batch, seq_length_batch = ru.generateSetFromTable(featuresTableBatch, n_steps, n_inputs)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, seq_length: seq_length_batch})
            acc_train = np.append(acc_train, accuracy.eval(feed_dict={X: X_batch, y: y_batch, seq_length: seq_length_batch}))
        acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val, seq_length: seq_length_val})
        print(epoch, "Train accuracy:", acc_train.mean(), "Validation accuracy:", acc_val)
