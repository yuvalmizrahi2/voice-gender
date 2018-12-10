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
    index = int(len(voice.index) * 0.7)
    data_x = np.array(voice.drop(['label'], axis=1))
    data_y = np.array(voice.loc[:, ['label']])
    print("finish split the data set")
    print("in the train have:" , index , "samples and in the test have:" , len(voice.index)-index , "samples")
    return data_x[:index], data_y[:index], data_x[index:], data_y[index:]


def model_training():
    print("start training the model")
    for i in range(0, 10000):
        sess.run(update, feed_dict={x: train_data_x, y_: train_data_y})
    print("finish training the model")
    loss2, _, acc = sess.run([loss, update, accuracy], feed_dict={
        x: train_data_x, y_: train_data_y})
    print("w:", sess.run(W), "\tb:", sess.run(b), "\tloss:",
          loss.eval(session=sess, feed_dict={x: train_data_x, y_: train_data_y})
          , "\tacc: " ,acc)


def model_testing():
    print("start testing the model")
    loss2, _, acc = sess.run([loss, update, accuracy], feed_dict={
        x: test_data_x, y_: test_data_y})
    print("Loss: " , loss2 ,  "\tAccuracy: ", acc)


voice = read_data_set()
train_data_x, train_data_y, test_data_x, test_data_y = split_data_set()
features = len(voice.columns) - 1
x =tf.placeholder(tf.float32,[None,features])
y_ =tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(x,W) + b)
loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_ ,logits=y)
loss = tf.reduce_mean(loss1)
update = tf.train.AdamOptimizer().minimize(loss)
prediction = tf.round(tf.sigmoid(tf.matmul(x,W) + b))
correct = tf.cast(tf.equal(prediction, y_), dtype=tf.float32)
accuracy = tf.reduce_mean(correct)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    model_training()
    model_testing()