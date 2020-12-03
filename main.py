import tensorflow as tf
import tkinter as tk

import numpy as np
import os

EPOCHS = 30
embedding_dim = 260
rnn_units = 200
BATCH_SIZE = 1
BUFFER_SIZE = 150
seq_length = 50

TEMP = 0.9
NUMGEN = 10000

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train(model, epochs, checkpoint_callback):
    return model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])

def generate_text(model, start_string, temperature, num_generate):
    # Evaluation step (generating text using the learned model)

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


text = open("Cards.txt", 'rb').read().decode(encoding='utf-8')
print('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
vocab_size = len(vocab)

# Length of the vocabulary in chars
vocab_size = len(vocab)

model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# sets up the UI Frames
master = tk.Tk()
master.title("Card Generator")
sliderFrame = tk.Frame(master)
sliderFrame.pack()

#Sets variables and generates text
def generateButtonFunc():
    TEMP = temperatureVar.get()
    NUMGEN = numGenVar.get()
    start = seedEntry.get()

    text = tk.Text(master, width=200)
    textToPrint = generate_text(model, start_string=start, temperature=TEMP, num_generate=NUMGEN)
    text.insert(tk.INSERT, textToPrint)
    text.pack()

def trainButtonFunc():
    EPOCHS = epochsVar.get()
    embedding_dim = embeddingDimVar.get()
    rnn_units = rnnUnitsVar.get()
    BATCH_SIZE = batchSizeVar.get()
    BUFFER_SIZE = bufferSizeVar.get()
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    vocab_size = len(vocab)

    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    if checkBtn.config('relief')[-1] == 'sunken':
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    history = train(model, EPOCHS, checkpoint_callback)

def toggle():
    if checkBtn.config('relief')[-1] == 'sunken':
        checkBtn.config(relief="raised")
    else:
        checkBtn.config(relief="sunken")

#Sets up the sliders and generate button

temperatureVar = tk.DoubleVar()
temperatureScale = tk.Scale(sliderFrame, from_=0.1, to=1, resolution=0.05, label="Temperature", variable = temperatureVar)
temperatureScale.pack(side=tk.LEFT)

numGenVar = tk.IntVar()
numGenScale = tk.Scale(sliderFrame, from_=1000, to=10000, resolution=100, label="Characters", variable = numGenVar)
numGenScale.pack(side=tk.LEFT)

epochsVar = tk.IntVar()
epochsScale = tk.Scale(sliderFrame, from_=1, to=50, resolution=1, label="Epochs", variable = epochsVar)
epochsScale.pack(side=tk.LEFT)

embeddingDimVar = tk.IntVar()
embeddingDimScale = tk.Scale(sliderFrame, from_=10, to=1000, resolution=10, label="Embedding Dimensions", variable = embeddingDimVar)
embeddingDimScale.pack(side=tk.LEFT)

rnnUnitsVar = tk.IntVar()
rnnUnitsScale = tk.Scale(sliderFrame, from_=1, to=1500, resolution=10, label="RNN Units", variable = rnnUnitsVar)
rnnUnitsScale.pack(side=tk.LEFT)

batchSizeVar = tk.IntVar()
batchSizeScale = tk.Scale(sliderFrame, from_=1, to=5, resolution=1, label="Batch Size", variable = batchSizeVar)
batchSizeScale.pack(side=tk.LEFT)

bufferSizeVar = tk.IntVar()
bufferSizeScale = tk.Scale(sliderFrame, from_=10, to=500, resolution=10, label="Buffer Size", variable = bufferSizeVar)
bufferSizeScale.pack(side=tk.LEFT)

entryLabel = tk.Label(master, text = "Generation Start")
entryLabel.pack()
seedEntry = tk.Entry(master)
seedEntry.insert(0, "Strike")
seedEntry.pack()
checkBtn = tk.Button(master, text="Load Checkpoints", relief="raised", command=toggle)
checkBtn.pack()

generateButton = tk.Button(master, command=generateButtonFunc, text="Generate")
generateButton.pack()

trainButton = tk.Button(master, command=trainButtonFunc, text="Train")
trainButton.pack()

tk.mainloop()