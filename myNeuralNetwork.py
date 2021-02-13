import numpy as np
import math
import os
import requests
import gzip
import io
from PIL import Image

# great source: http://neuralnetworksanddeeplearning.com/chap1.html
# this was more of a research project than anything

def loadDataset():
    # not working atm
    # for now I'm downloading and putting the files in the folder manually
    def download(filename):
        print ("Downloading " + filename)

        r = requests.get('http://yan.lecun.com/exdb/mnist/' + filename)
        cf = io.StringIO(r.content)
        print("hi"+cf)

        # import urllib.request
        with open(filename, 'wb') as f:
            r = requests.get('http://yan.lecun.com/exdb/mnist/' + filename)
            f.write(r.content)

        print ("\nSuccessful download.")

    '''returns a list of matricies of numbers with values between 0 and 1
    values in the matrix are the grayscale values of the pixels of an image
    matricies are 28x28'''
    def loadImages(filename):

        with gzip.open(filename, 'rb') as f:
            # data is a 1d array of bytes
            data = np.frombuffer(f.read(), np.uint8, offset=16)

            '''creates an array of image matrixes
            first parameter determines the number of images,
              -1 means to use the length of data and
              the other dimensions to calculate it automatically
            second is the number of channels, 1 because it's monochrome,
              would be 3 if it were RGB for example
            third and fourth are the size of the image'''
            data = data.reshape(-1, 1, 28, 28)

            # convert from bytes to floats
            data = data/np.float32(256)
            return data

    # returns a list of ints between 0 and 9,
    # each being the value of an image corresponding by index
    def loadLabels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    trainIms = loadImages('train-images-idx3-ubyte.gz')
    trainLbl = loadLabels('train-labels-idx1-ubyte.gz')
    testIms = loadImages('t10k-images-idx3-ubyte.gz')
    testLbl = loadLabels('t10k-labels-idx1-ubyte.gz')
    return trainIms, trainLbl, testIms, testLbl

trainIms, trainLbl, testIms, testLbl = loadDataset()


# show an image and print it's matrix
'''
import matplotlib.pyplot as plt
plt.imshow(trainIms[0][0], cmap = "gray")
plt.show()
print(trainIms[0][0])
'''


def sigmoid(z):
    '''the sigmoid function takes any number and outputs a number between 0 and 1
    -inf would ouput 0, inf would output 1, 0 outputs .5
    gets pretty close to 1/0 at +-4'''
    return 1/(math.exp(-1*z) + 1)

def sigmoidPrime(z):
    return sigmoid(z)*(1 - sigmoid(z))


class Network(object):
    '''sizes is a list of numbers,
    each number is the size of each layer of the network
    ex Network([3,2,1]) has 3 neurons in the input layer,
    2 in a hidden layer, and 1 in an output layer'''
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        '''in each layer but the first, each neuron has one bias
        biases in Network(3,4,5,2) example
        where a1, a2, b1, b2 etc are random numbers
        biases = [[a1,a2,a3,a4]
                 [b1,b2,b3,b4,b5]
                 [c1,c2]]'''
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # each neuron has an array of weights with equal length to the size
        # of the previous layer
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        '''the actual values in weights and biases are random standard
        normal distribution, aka relatively close to 0
        the specific values don't matter yet, they just needs to matter
        eventually via learning'''

    def feedforward(self, a):
        '''input is an array of values of length equal to previous layer of
        values between 0 and 1
        multiplies each value in the array by each associated weight and sums
        the products aka dot product, then adds the bias
        put that value through the sigmoid function and you have an output
        between 0 and 1
        '''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    '''SDG - stochastic gradient descent
    Training method.
    training_data is a list of tuples(x,y),
    x is a matrix representing an image, y is the desired output
    an epoch is one learning iteration, 'epochs' is the number of iterations
    mini_batch_size is PUT SOMETHING HERE
    eta is the learning rate PUT SOMETHING HERE
    test_data is usually a smaller, different version of training data that
    can be supplied to print out the accuracy for each epoch based on the
    test data'''
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
            # first, shuffle the data and partition it into minibatches
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            # then, apply one step of sgd with each batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # print stuff if you want
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    '''Where the actual sdg happens.
    Mini_batch is a subset of training data, it's still a list of
    tuples(x,y) where x is the image and y is the label
    eta is still the learning rate'''
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb
                       for b, nb ni zip(self.biases, nabla_b)]
