# Streaming metrics add two local variables total and count. You can find and reset them to get the required behavior.

import tensorflow as tf

value = 0.1
with tf.name_scope('foo'):
    mean_value, update_op = tf.contrib.metrics.streaming_mean(value)

init_op = [tf.initialize_variables(tf.local_variables())]
stream_vars = [i for i in tf.local_variables() if i.name.split('/')[0] == 'foo']
reset_op = [tf.initialize_variables(stream_vars)]
with tf.Session() as sess:
    sess.run(init_op)
    for j in range(3):
        for i in range(9):
            _, total, count = sess.run([update_op] + stream_vars)
            mean_val = sess.run([mean_value])
            print(total, count, mean_val)
        sess.run(reset_op)
        print('')
