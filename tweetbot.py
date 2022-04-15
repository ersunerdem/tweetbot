import numpy
import sys
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import random

#Following tutorial from: https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/

nltk.download('all')

#File name for the training data
training_fn = "training_data.txt"
#File name for test data
test_fn = "test_data.txt"

#Encoding in open() set to "mcbs" due to following UnicodeDecodeError:
#UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 33234: character maps to <undefined>
training_file = open(training_fn, encoding="mbcs").read()

def tokenize_words(input):
    input = input.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)



processed_inputs = tokenize_words(training_file)
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))

print(f"Total number of characters: {len(processed_inputs)}")
print(f"Total vocabulary: {len(chars)}")

seq_length = random.randrange(100, 200, step=1)
print(f"This sequence length: {seq_length}")
x_data = []
y_data = []

for i in range(0, len(processed_inputs) - seq_length, 1):
    #Input and Output Sequences
    #Input Sequence: current character plus sequence length
    in_seq = processed_inputs[i:i + seq_length]
    #Output sequence: initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    #Convert list of characters to integers and add values to list
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

print(f"Total Patterns: {len(x_data)}")

X = numpy.reshape(x_data, (len(x_data), seq_length, 1))
X = X/float(len(chars))

y = np_utils.to_categorical(y_data)

#Model

#This is the model from the article/tutorial... maybe play with the shape/design of it??
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#Compile, set for saving weights
model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

#Train for 40 epochs
model.fit(X, y, epochs=40, batch_size=64, callbacks=desired_callbacks)

#Once trained, load weights and recompile with saved weights
model.load_weights(filepath)
model.compile(loss='categorical_crossentropy', optimizer='adam')

num_to_char = dict((i, c) for i, c in enumerate(chars))


#Character generation
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

#Convert random seed to float values, predict next character and generate text
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
