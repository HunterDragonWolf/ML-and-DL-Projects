import random
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# Importing/Getting the Shakespeare's write-up from a link using tensorflow
filepath = tf.keras.utils.get_file('Shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Getting the text from the file into the script
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() # get_file() returns to open(), rb = read binary mode, read() to get the text.
                                                                    # decode() to decode, converting into lowercase to inrease accurace(model predicts the next character)

# Getting text and converting into numerical format(for the neural network)
# so that we can use the numpy array into the network.

characters = sorted(set(text))  # Set of all the characters that are sorted

# Creating 2 dictionaries to convert the characters into numeric format and vice versa

char_to_index = dict((c, i) for i, c in enumerate(characters))   #enumerate asigns a numbers to each character
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40         # Using 40 characters as feature data
STEP_SIZE = 3           # Number of characters to shift

'''
sentences = []          # The features
next_characters = []    # The target

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)   # Whenever in a specific sentence at a specific position a certain character occurs is set to true/1
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

# Filling the arrays with for loops
# Running one for loop over all the sentences, taking all the sentences we created,
# all the sequences and assigning an index to them. For each sentence we enumerate every character in the sentence.

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Building the Neural Network (RNN):

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))    # softmax = scales the output so that all the values add up to 1
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=3)
model.save('Shakespeare text Generator.h5')
'''

model = tf.keras.models.load_model('Shakespeare text Generator.h5')

# To make generate texts
        # This function takes the predictions of the model and picks one character.

def sample(preds, temperature=1.0):                 # Depending on temperature; conservative or experimental(higher temp=more creative sentences)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_texts(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

pickle.dump(model, open('textgenerator.pkl','wb'))
Model = pickle.load(open('textgenerator.pkl','rb'))

# Results

print('---------0.6---------')
print(generate_texts(250, 0.6))

