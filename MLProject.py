import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
import numpy as np
K.clear_session()

nameSize = 4
costSize = 1
typeSize = 1
textSize = 50
outputNeurons = costSize+nameSize+typeSize+textSize
wordDict = {"": 0}
revWordDict = {0: ""}
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

def readInputFile(fileName):
    outputList = [];
    file = open(fileName, "r")
    for line in file:
        outputList.append(line[:len(line)-1])#the -1 gets rid of newline characters
    return outputList

def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(outputNeurons, input_shape=(100,), activation = "relu", use_bias=False))
    model.add(layers.Dense(outputNeurons, activation = "relu", use_bias=False))
    model.add(layers.Dense(outputNeurons, activation = "relu", use_bias=False))
    return model

def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(50, input_shape=(outputNeurons,), activation = "relu", use_bias=False))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(50, activation = "relu", use_bias=False))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation = "relu", use_bias=False))
    return model

def addWordsToDictionaryFromString(line):
    catagories = line.split(";") #Separates by ;'s and gets rid of ;'s
    
    for catagory in catagories:
        
        words = catagory.split() #Separates by spaces 
        
        for word in words:

            #Finds errors where there isn't a space after a period
            if word.find(".") >-1 and not word[len(word) -1] == ".":
                print(word)
                
            #If there is a period, add it to the dictionary
            if word[len(word) -1] == ".":
                if not "." in wordDict:
                    wordDict["."] = len(wordDict)
                    revWordDict[len(wordDict)-1] = "."
                word = word[len(word) -1]

            #Add word to dictionary
            if not word in wordDict:
                wordDict[word.lower()] = len(wordDict)
                revWordDict[len(wordDict)-1] = word.lower()


def getWordsFromNumbers(numList):
    word = "Name: "
    count = 0
    #Iterate through the numbers
    for num in numList:

        if count == 3:
            word += " Cost: "
        if count == 4:
            word += " Type: "
        if count == 5:
            word += " Description: "
        
        #If the word to add is a period, remove the trailing period
        if revWordDict[num] == ".":
            word = word[: len(word) -1]
        
        word += revWordDict[num]

        #If the word was not empty, add a space
        if not revWordDict[num] == "":
            word += " "

        
        count += 1
    return word

def classifier_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


    

cardList = readInputFile("Cards.txt")
for card in cardList:
    addWordsToDictionaryFromString(card)


generator = generator_model()

print(wordDict)



