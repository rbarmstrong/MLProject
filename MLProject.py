from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras import backend as K
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

costSize = 1
nameSize = 4
typeSize = 1
textSize = 50


def readInputFile(fileName):
    outputList = [];
    file = open(fileName, "r")
    for line in file:
        outputList.append(line[:len(line)-1])#the -1 gets rid of newline characters
    return outputList

def generator_model():
    outputNeurons = costSize+nameSize+typeSize+textSize
    model = tf.keras.Sequential()
    model.add(layers.Dense(outputNeurons, input_shape=(100,), activation = "relu", use_bias=False))
    model.add(layers.Dense(outputNeurons, activation = "relu"))
    model.add(layers.Dense(outputNeurons, activation = "relu"))
    return model