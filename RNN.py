import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os
import sys
import random


# load the text and convert everything to lowercase 
filepath = "/home/student/Documents/CS155/MiniProject3/shakespeare.txt"
text = open(os.path.join(os.getcwd(), "/home/student/Documents/CS155/MiniProject3/shakespeare.txt")).read()
poem = []
for i in text: 
    if not i.isdigit():
        poem.append(i)
poem = ''.join(poem)
poem = poem.lower()

# create mapping of chars to inetegers and integers to chars
chars = sorted(list(set(poem)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summary of data 
num_chars = len(poem)
num_vocab = len(chars)
print("Total Characters: ", num_chars)
print("Total Vocab: ", num_vocab)

# creation of input and output pairs as integers 
sequence_len = 40
Xdata = []
Ydata = []
for i in range(0, num_chars - sequence_len, 1): 
    seq_in = poem[i:i + sequence_len]
    seq_out = poem[i + sequence_len]
    Xdata.append([char_to_int[c] for c in seq_in])
    Ydata.append(char_to_int[seq_out])
num_patterns = len(Xdata)
print("Total patterns: ", num_patterns)

# reshape Xdata 
X = np.reshape(Xdata, (num_patterns, sequence_len, 1))
# normalize X
X = X / float(num_vocab)
# one hot encode Y 
y = np_utils.to_categorical(Ydata)

# LSTM model 
model = Sequential()
model.add(LSTM(100, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Lambda(lambda x: x / 0.25))
model.add(Dense(y.shape[1], activation = 'softmax'))
'''
# load network weights 
filename = "weights-improvement-33-2.1916.hdf5"
model.load_weights(filename)'''
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# checkpoint 
path = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 1, save_best_only = True)
callbacks_list = [checkpoint]
# fit the model 
model.fit(X, y, epochs = 40, batch_size = 32, callbacks = callbacks_list)


'''
# pick a random seed 
#start_seed = np.random.randint(0, len(Xdata) - 1)
#pattern = Xdata[start_seed]
print("Seed: ")
#print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
seed = "shall i compare thee to a summer's day?\n"
print(seed)
pattern = [char_to_int[c] for c in seed]

# generate characters 
for i in range(500):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(num_vocab)
    prediction = model.predict(x, verbose = 0)
    randomword = random.randint(0,len(prediction[0])-1)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    
    
    
print("\n")
print("Sample Poem Seed: ")
print("----------------- ")
print("\n")
for i in range(14): 
    start_seed = np.random.randint(0, len(Xdata) - 1)
    pattern = Xdata[start_seed]
    print(''.join([int_to_char[value] for value in pattern]))

# fit the model 
model.fit(X, y, epochs = 1, batch_size = 128, callbacks = callbacks_list)
'''
