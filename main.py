import tensorflow as tf
import tkinter as tk

import numpy as np
import os

EPOCHS = 1
embedding_dim = 256
rnn_units = 1024
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

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

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

#history = train(model, EPOCHS, checkpoint_callback)

# sets up the UI Frames
master = tk.Tk()
master.title("Card Generator")
sliderFrame = tk.Frame(master)
sliderFrame.pack()

#Sets variables and generates text
def generateButtonFunc():
    TEMP = temperatureVar.get()
    NUMGEN = numGenVar.get()
    text = tk.Text(master, width=200)
    textToPrint = generate_text(model, start_string=u"Strike", temperature=TEMP, num_generate=NUMGEN)
    text.insert(tk.INSERT, textToPrint)
    text.pack()
    

#Sets up the sliders and generate button

temperatureVar = tk.DoubleVar()
temperatureScale = tk.Scale(sliderFrame, from_=0.1, to=1, resolution=0.05, label="Temperature", variable = temperatureVar)
temperatureScale.pack(side=tk.LEFT)

numGenVar = tk.IntVar()
numGenScale = tk.Scale(sliderFrame, from_=1000, to=10000, resolution=100, label="characters", variable = numGenVar)
numGenScale.pack(side=tk.LEFT)


generateButton = tk.Button(master, command=generateButtonFunc, text="generate")
generateButton.pack()



