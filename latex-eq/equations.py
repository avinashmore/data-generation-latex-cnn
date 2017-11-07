from cStringIO import StringIO
import matplotlib.pyplot as plt
import string
import random
import os
from PIL import Image
import glob
import numpy

letters = tuple([i for i in string.ascii_lowercase[0:26]])
numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
operation = ('frac', 'int','^', 'frac', 'sum','int', '^', 'sum')
plt.rcParams['font.sans-serif'] = "Comic Sans MS"
plt.rcParams['font.family'] = "sans-serif"


def render_latex(formula, fontsize=10, dpi=300, format_='svg'):
    fig = plt.figure(figsize=(1, 0.5))
    fig.text(0.1, 0.5, u'${}$'.format(formula), fontsize=fontsize)
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
            expression = r'\frac' + '{' + letter1 + '}' + '{' + letter2 + '}'
        elif operation[op] == '^':
            expression = r'' + letter1 + '^' + letter2
        elif operation[op] == 'sum':
            number = numbers[random.randint(0, len(numbers) - 1)]
            expression = r'\sum_{' + letter1 + '=1}^{\infty}' + number + '^{' + letter1 + '}'
        elif operation[op] =='int':
            expression = r'\int_{' + letter1 +'} ^ {'+ letter2 + '}' + letter3 +'^ 2d' + letter3
        image_bytes = render_latex(expression, fontsize=10, dpi=200, format_='png')
        image_name = './data/' + str(i) + '.png'
        with open(image_name, 'wb') as image_file:
            image_file.write(image_bytes)


def read_data():
    imageFolderPath = 'data'
    imagePath = glob.glob(imageFolderPath + '/*.png')
    im_array = numpy.array([numpy.array(Image.open(img).convert('L'), 'f') for img in imagePath])
    return im_array


if __name__ == '__main__':
    generate_data(20)
    input_data = read_data()
