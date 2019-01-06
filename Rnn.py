import numpy as np
import pandas as pd
import tensorflow as tf


def read_data_set():
    print("start reading the data set")
    data_set = pd.read_csv('voice-gender dataset.csv')
    data_set.label = data_set.label.replace(to_replace=['male', 'female'], value=[0, 1])
    data_set = data_set.sample(frac=1).reset_index(drop=True)
    print("finish reading the data set")
    return data_set


def split_data_set():
    print("start split the data set to 70-30")
    index = int(211 * 0.7)
    data_x = np.array(voice.drop(['label'], axis=1)).reshape(211, 15, 20, 1)
    data_y = np.array(voice.loc[:, ['label']])
    data_y = np.insert(data_y, 1, 1 - data_y[:, 0], axis=1)
    data_y=data_y.reshape(211,15,2)
    print("in the train have:", index, "samples and in the test have:", 211 - index, "samples")
    return data_x[:index], data_y[:index], data_x[index:], data_y[index:]


voice = read_data_set()
train_data_x, train_data_y, test_data_x, test_data_y = split_data_set()

possible_label = 2
num_past_features = 20
num_of_epochs = 200
cellsize = 30

x = tf.placeholder(tf.float32, [None, num_past_features, 1])
y = tf.placeholder(tf.float32, [None, possible_label])

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cellsize, forget_bias=1.0)
output, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

output = tf.transpose(output, [1, 0, 2])
last = output[-1]

W = tf.Variable(tf.truncated_normal([cellsize, possible_label], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[possible_label]))

z = tf.matmul(last, W) + b
res = tf.nn.softmax(z)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(res), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(res,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

acc_trace = tf.summary.scalar('accuracy', accuracy)
loss_trace = tf.summary.scalar('loss', cross_entropy)

with tf.Session() as sess:
    file_writer1 = tf.summary.FileWriter('RNN/train', sess.graph)
    file_writer2 = tf.summary.FileWriter('RNN/test', sess.graph)
    file_writer3 = tf.summary.FileWriter('RNN/loss', sess.graph)
    sess.run(init)
    for ephoch in range(num_of_epochs):
        acc = 0
        tr=0
        for i in range(147):
            batch_xs=train_data_x[i]
            batch_ys=train_data_y[i]
            _, curr_loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y: batch_ys})
            acc += accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
            tr = tr+1
        print("step %d, loss %g, training accuracy %g"%(ephoch, curr_loss, acc/tr))
    acc = 0
    tr = 0
    for i in range(64):
        batch_xs = test_data_x[i]
        batch_ys = test_data_y[i]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc += accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
        tr = tr + 1
    print("test accuracy %g"%(acc/tr))
