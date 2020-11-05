import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
import numpy as np
import time
import os
K.clear_session()
tf.config.experimental.list_physical_devices('GPU')

nameSize = 4
costSize = 1
typeSize = 1
textSize = 30
outputNeurons = costSize+nameSize+typeSize+textSize
wordDict = {"": 0}
revWordDict = {0: ""}
input_data = [[]]
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
EPOCHS = 1000
BATCH_SIZE = 1
noise_dim = 100
num_examples_to_generate = 16
generator_optimizer = keras.optimizers.Adam()
discriminator_optimizer = keras.optimizers.Adam()

def readInputFile(fileName):
    outputList = [];
    file = open(fileName, "r")
    for line in file:
        outputList.append(line[:len(line)-1])#the -1 gets rid of newline characters
    return outputList

def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(100, input_shape=(noise_dim,), activation = "relu", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(100, activation = "tanh"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(outputNeurons, activation = "relu"))
    return model

def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(100, input_shape=(outputNeurons,), activation = "relu", use_bias=False))
    model.add(layers.Dropout(0.5))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation="sigmoid",  use_bias=False))
    model.add(layers.Dropout(0.5))
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
                    
                word = word[:len(word) -1]

            #Add word to dictionary
            if not word.lower() in wordDict:
                
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
        if num in revWordDict:
            #If the word to add is a period, remove the trailing period
            if revWordDict[num] == ".":
                word = word[:len(word) -1]
            
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

def train_step(cards):
    noise = tf.random.uniform(shape = [BATCH_SIZE,noise_dim], maxval = len(wordDict), dtype = tf.int32)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_cards = generator(noise, training=True)

        real_output = discriminator(cards, training=True)
        fake_output = discriminator(generated_cards, training=True)

        gen_loss = generator_loss(fake_output)
        class_loss = classifier_loss(real_output, fake_output)

        #printTest = "gen: " + str(gen_loss) + " class: " + str(class_loss)
        #print(printTest)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_classifier = disc_tape.gradient(class_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_classifier, discriminator.trainable_variables))

def generateSaveCards(model, test_input, epoch):
    cards = model(test_input, training=False)
    f = open("outputFile.txt", "w")
    f.write("Card at epoch {}".format(epoch))
    f.write(getWordsFromNumbers(tf.make_ndarray(tf.make_tensor_proto(cards))[0].round()))
    f.close()

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for cards in dataset:
            train_step(tf.convert_to_tensor(cards))

        noise = tf.random.uniform(shape=[1, noise_dim], maxval=len(wordDict), dtype=tf.int32)
        generateSaveCards(generator, noise, epoch + 1)
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    generateSaveCards(generator, noise, epochs)


cardList = readInputFile("Cards.txt")
for card in cardList:
    addWordsToDictionaryFromString(card)


train_data = [[]]
i = 0
for card in cardList:
    train_data.append([])
    categories = card.split(";") #Separates by ;'s and gets rid of ;'s
    for j in range(len(categories)):
        words = categories[j].split() #Separates by spaces
        periodCount = 0
        for word in words:
            if word[len(word) -1] == ".":
                periodCount += 1
                train_data[i].append(wordDict[word[:len(word) -1].lower()])
                train_data[i].append(wordDict["."])
            else:
                train_data[i].append(wordDict[word.lower()])

        neededInputCount = 0
        if j == 0:
            neededInputCount = nameSize
        elif j == 1:
            neededInputCount = costSize
        elif j == 2:
            neededInputCount = typeSize
        elif j == 3:
            neededInputCount = textSize
        
        for k in range(len(words) + periodCount, neededInputCount):
            train_data[i].append(wordDict[""])
    i += 1
train_data = train_data[:len(train_data)-1]

generator = generator_model()
discriminator = discriminator_model()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

train(train_data, EPOCHS)

