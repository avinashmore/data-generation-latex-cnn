import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

print(mnist)

# change to 40
input_size = 784
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# matrix height x Width
# squash out the input value of 28x28 into input
X = tf.placeholder(tf.float32, [None, input_size])
Y = tf.placeholder(tf.float32)


# (input_data * weights) + biases
# we need bias in teh case where all the input data is 0, no neuron wouold fire on relu. so we can have bias to get some neurons to fire.
# can make the network overall to be dynamic

def model(data):
    hl1 = {'weights': tf.Variable(tf.random_normal([input_size, n_nodes_hl1])),
           'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hl2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
           'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hl3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
           'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases
    l1 = tf.nn.relu(tf.add(tf.matmul(data, hl1['weights']), hl1['biases']))
    l2 = tf.nn.relu(tf.add(tf.matmul(l1, hl2['weights']), hl2['biases']))
    l3 = tf.nn.relu(tf.add(tf.matmul(l2, hl3['weights']), hl3['biases']))

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


# takes input data x
def train_network(x, y):
    prediction = model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    n_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(0, n_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples / batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            print('Epoch', epoch, 'completed out of', n_epochs)

            # assert if equal. check the one hot of the prediction to the vector of the labeled data.
            # argmax is gonna return index of maximum value. we want these indexes to be the same


        # test trained model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_network(X, Y)
