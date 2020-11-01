import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
K.clear_session()


costSize = 1
nameSize = 4
typeSize = 1
textSize = 50
wordDict = {"": 0}

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

def addWordsToDictionaryFromString(line):
    categories = line.split(";") #Separates by ;'s and gets rid of ;'s
    
    for category in categories:
        
        words = category.split() #Separates by spaces 
        
        for word in words:

            #Finds errors where there isn't a space after a period
            if word.find(".") >-1 and not word[len(word) -1] == ".":
                print(word)
                
            #If there is a period, add it to the dictionary
            if word[len(word) -1] == "." and not "." in wordDict:
                wordDict["."] = len(wordDict)

            #Add word to dictionary
            if not word in wordDict:
                wordDict[word.lower()] = len(wordDict)


cardList = readInputFile("Cards.txt")
for card in cardList:
    addWordsToDictionaryFromString(card)

print(wordDict)

