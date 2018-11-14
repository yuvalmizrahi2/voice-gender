import numpy as np
import pandas as pd
import tensorflow as tf

voice = pd.read_csv('voice.csv')
voice.label = voice.label.replace(to_replace=['male', 'female'], value=[0, 1])
voice = voice.sample(frac=1).reset_index(drop=True)

features = len(voice.columns) - 1
index = int(len(voice.index) * 0.7)

data_x = np.array(voice.drop(['label'] , axis=1))
data_y = np.array(voice.loc[:,['label']])

train_data_x = data_x[:index]
train_data_y = data_y[:index]
test_data_x = data_x[index:]
test_data_y = data_y[index:]

x =tf.placeholder(tf.float32,[None,features])
y_ =tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(x,W) + b)
loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_ ,logits=y)
loss = tf.reduce_mean(loss1)
update = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,1000000):
    sess.run(update,feed_dict={x:train_data_x , y_:train_data_y})
print("w:", sess.run(W) , " b:" , sess.run(b) , " loss:" , loss.eval(session=sess , feed_dict = {x:train_data_x , y_:train_data_y}))
count = 0
for i in range(test_data_x.shape[0]):
    temp = sess.run(tf.nn.sigmoid(((np.matmul(np.array([test_data_x[i]]), sess.run(W))) + sess.run(b))[0][0]))
    num = test_data_y[i][0]
    if (temp < 0.5 and num == 0) or (temp >=0.5 and num == 1):
        count = count +1
print(count/test_data_x.shape[0])
