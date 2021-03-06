from keras.datasets import imdb		# the data
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout


# Set the vocabulary size and load in training and test data.
vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
# print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))


# Loaded dataset with 25000 training samples, 25000 test samples
# Inspect a sample review and its label.
# print('---review---')
# print(X_train[6])
# print('---label---')
# print(y_train[6])


# map the review back to the original words.
# word2id = imdb.get_word_index()
# id2word = {i: word for word, i in word2id.items()}
# print('---review with words---')
# print([id2word.get(i, ' ') for i in X_train[1]])
# print('---label---')
# print(y_train[1])


# # Maximum review length and minimum review length.
# print('Maximum review length: {}'.format(
# len(max((X_train + X_test), key=len))))


# print('Minimum review length: {}'.format(
# len(min((X_test + X_test), key=len))))


# Pad sequences
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


# RNN model for sentiment analysis
embedding_size=32
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
# print(model.summary())


# Train and evaluate our model
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])


# Training takes a while... a lot of it
batch_size = 64
num_epochs = 3
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)


# scores[1] will correspond to accuracy if we pass metrics=[‘accuracy’]
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])