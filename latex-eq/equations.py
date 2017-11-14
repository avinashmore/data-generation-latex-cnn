
from cStringIO import StringIO

import matplotlib.pyplot as plt
import string
import random
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
import pickle

letters = tuple([i for i in string.ascii_lowercase[0:26]])
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
operation = ('frac', 'int', '^', 'frac', 'sum', 'int', '^', 'sum')
plt.rcParams['font.sans-serif'] = "Comic Sans MS"
plt.rcParams['font.family'] = "sans-serif"
data_size = 20000
test_data_size = 200
y = np.empty((data_size, 4), dtype=int)


def render_latex(formula, fontsize=10, dpi=300, format_='svg'):
    fig = plt.figure(figsize=(0.3, 0.25))
    fig.text(0.05, 0.35, u'${}$'.format(formula), fontsize=fontsize)
    buffer_ = StringIO()
    fig.savefig(buffer_, dpi=dpi, transparent=True, format=format_, pad_inches=0.0)
    plt.close(fig)
    return buffer_.getvalue()


def generate_data(image_count):
    for i in range(0, image_count):
        op = random.randint(0, len(operation) - 1)
        letter1 = letters[random.randint(0, len(letters) - 1)]
        letter2 = letters[random.randint(0, len(letters) - 1)]
        letter3 = letters[random.randint(0, len(letters) - 1)]

        if operation[op] == 'frac':
            y[i] = [1, 0, 0, 0]
            expression = r'\frac' + '{' + letter1 + '}' + '{' + letter2 + '}'
        elif operation[op] == '^':
            y[i] = [0, 1, 0, 0]
            expression = r'' + letter1 + '^' + letter2
        elif operation[op] == 'sum':
            y[i] = [0, 0, 1, 0]
            number = numbers[random.randint(0, len(numbers) - 1)]
            expression = r'\sum_{' + letter1 + '=1}^{\infty}' + number + '^{' + letter1 + '}'
        elif operation[op] == 'int':
            y[i] = [0, 0, 0, 1]
            expression = r'\int_{' + letter1 + '} ^ {' + letter2 + '}' + letter3 + '^ 2d' + letter3
        image_bytes = render_latex(expression, fontsize=5, dpi=200, format_='png')
        image_name = './data/' + str(i) + '.png'
        with open(image_name, 'wb') as image_file:
            image_file.write(image_bytes)

    pickle.dump(y, open("save.p", "wb"))



def read_data():
    image_folder_path = 'data'
    image_path = glob.glob(image_folder_path + '/*.png')

    
    im_array = np.array([np.reshape(np.array(Image.open(img).convert('L'), 'f'),(3000)) for img in image_path])
    # print im_array.shape
    y_from_file = pickle.load(open("save.p", "rb"))
    return im_array, y_from_file


def next_batch(n, input_data):
    x_batch = np.empty((n, 3000), dtype=int)
    y_batch = np.empty((n, 4), dtype=int)

    for i in range(0, n):
        number = random.randint(0, data_size - 1)
        input_data_point = input_data[number]
        output_data_point = y[number]
        # print sess.run(output_data_point)
        # print "type", type(input_data_point)
        # print input_data_point
        x_batch[i] = input_data_point
        y_batch[i] = output_data_point

    return x_batch, y_batch


if __name__ == '__main__':
    # generate_data(data_size)
    input_data, y_from_file = read_data()
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 3000])
    y_ = tf.placeholder(tf.float32, shape=[None, 4])
    W = tf.Variable(tf.zeros([3000, 4]))
    b = tf.Variable(tf.zeros([4]))
    sess.run(tf.global_variables_initializer())
    xy = tf.matmul(x, W) + b

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=xy))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    chunk = 0
    for _ in range(1000):
        batch_x, batch_y = next_batch(100, input_data)
        train_step.run(feed_dict={x: batch_x, y_: batch_y})

    correct_prediction = tf.equal(tf.argmax(xy, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch_test, batch_lables = next_batch(test_data_size, input_data)
    print(accuracy.eval(feed_dict={x: batch_test, y_: batch_lables}))