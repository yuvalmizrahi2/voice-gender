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
    print("in the train have:", index, "samples and in the test have:", len(voice.index)-index, "samples")
    return data_x[:index], data_y[:index], data_x[index:], data_y[index:]


voice = read_data_set()
train_data_x, train_data_y, test_data_x, test_data_y = split_data_set()
features = len(voice.columns) - 1
x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
hidden_layers = 1
hidden_layer_neurons = [10]
if hidden_layers > 0:
    W = tf.Variable(tf.truncated_normal([features, hidden_layer_nodes[0]], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes[0]]))
    z = tf.add(tf.matmul(x, W), b)
    a = tf.nn.relu(z)
    for W_num in range(1, hidden_layers + 1):
        if W_num == hidden_layers:
            W = tf.Variable(tf.truncated_normal([hidden_layer_nodes[W_num - 1], 1], stddev=0.1))
            b = tf.Variable(0.)
            z = tf.matmul(a, W) + b
        else:
            W = tf.Variable(tf.truncated_normal([hidden_layer_nodes[W_num - 1], hidden_layer_nodes[W_num]], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes[W_num]]))
            z = tf.add(tf.matmul(a, W), b)
            a = tf.nn.relu(z)
