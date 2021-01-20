import numpy as np
import math
import os
import requests
import gzip
import io
from PIL import Image

#where I learned most of this: http://neuralnetworksanddeeplearning.com/chap1.html
#I love free textbooks

def loadDataset():
    def download(filename):
        #not working atm, I'm just downloading it from a web browser and dropping it in the folder
        print ("Downloading " + filename)

        r = requests.get('http://yan.lecun.com/exdb/mnist/' + filename)
        cf = io.StringIO(r.content)
        print("hi"+cf)

        
        #import urllib.request
        with open(filename, 'wb') as f:
            r = requests.get('http://yan.lecun.com/exdb/mnist/' + filename)
            f.write(r.content)
        
        print ("\nSuccessful download.")

    def loadImages(filename):
        
        with gzip.open(filename, 'rb') as f:
            #data is a 1d array of bytes
            data = np.frombuffer(f.read(), np.uint8, offset = 16)

            #creates an array of image matrixes
            #first parameter determines the number of images, -1 means to use the length of data and
            #the other dimensions to calculate it automatically
            #second is the number of channels, 1 because it's monochrome, would be 3 if it were RGB for example
            #third and fourth are the size of the image
            data = data.reshape(-1, 1, 28, 28)

            #convert from bytes to floats
            data = data/np.float32(256)
            return data

    def loadLabels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset = 8)
            #data is now a 1d array of ints, the integer associated with each image corresponding by order

        return data

    trainIms = loadImages('train-images-idx3-ubyte.gz')
    trainLbl = loadLabels('train-labels-idx1-ubyte.gz')
    testIms = loadImages('t10k-images-idx3-ubyte.gz')
    testLbl = loadLabels('t10k-labels-idx1-ubyte.gz')
    return trainIms, trainLbl, testIms, testLbl

trainIms, trainLbl, testIms, testLbl = loadDataset()

#show an image, not permanent
import matplotlib.pyplot as plt
plt.imshow(trainIms[0][0], cmap = "gray")
plt.show()
print(trainIms[0][0])

#the sigmoid function takes any number and outputs a number between 0 and 1
#-inf would ouput 0, inf would output 1, 0 outputs .5
#gets pretty close to 1/0 at +-4
def sigmoid(z):
    return 1/(math.exp(-1*z) + 1)

