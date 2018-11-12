import numpy as np
import pandas as pd
import tensorflow as tf

voice = pd.read_csv('voice.csv')
voice.label = voice.label.replace(to_replace=['male', 'female'], value=[0, 1])
voice = voice.sample(frac=1).reset_index(drop=True)

features = len(voice.columns) - 1

data_x = np.array(voice.drop(['label'] , axis=1))
data_y = np.array(voice.loc[:,['label']])

x =tf.placeholder(tf.float32,[None,features])
y_ =tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(x,W) + b)
loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y ,logits=y_)
loss = tf.reduce_mean(loss1)
update = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,100000):
    sess.run(update,feed_dict={x:data_x , y_:data_y})
    if i % 10000 == 0:
        print('Iteration:' , i , ' W:' , sess.run(W) , ' b:' , sess.run(b))