import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


df = pd.read_csv('iris.data.txt', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
df = df.replace({'class': {'Iris-setosa': 3, 'Iris-versicolor': 5, 'Iris-virginica': 7}})
# df = df.replace('Iris-setosa', 3)
# df = df.replace('Iris-versicolor', 5)
# df = df.replace('Iris-virginica', 7)
df['setosa'] = (df['class'] % 5.0) % 2.0
df['versicolor'] = ((df['class'] % 7.0) % 3.0) / 2
df['virginica'] = (df['class'] % 3.0) % 2.0

X = np.array(df.drop(['class', 'setosa', 'versicolor', 'virginica'], 1))
X = preprocessing.scale(X)
y = np.array(df[['setosa', 'versicolor', 'virginica']])
# y = preprocessing.scale(y)

train_x, test_x, train_y, test_y = model_selection.train_test_split(X, y, test_size=0.2)

n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50
n_classes = 3
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum': n_nodes_hl3,
                  'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum': None,
                'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}


def neural_network_model(data):

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output


def train_neural_network(x):
    predict = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=y))
    optimizer = tf.train.AdadeltaOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start: end])
                batch_y = np.array(train_y[start: end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)
