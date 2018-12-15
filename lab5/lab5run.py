from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing import sequence
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

all_files = ["imdb_labelled.txt","amazon_cells_labelled.txt","yelp_labelled.txt"]
dataframe = pd.concat(pd.read_csv(file, sep='\t', names = ['txt', 'label'],index_col=None, header=0) for file in all_files)
vocabulary_size = len(dataframe)

# Normalisation of text is done here
norm_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=set(stopwords.words('english')))

y = dataframe.label    #setting dependent variables, the labels 0 for negative and 1 for positive sentiment
x = norm_vectorizer.fit_transform(dataframe.txt)    #transforming data in the dataframe to features from text

# Training testing split
# Using a random state to guarantee the same results whenever training is done
x_training, x_testing, y_training, y_testing = train_test_split(x, y, random_state=29, train_size = 0.95, test_size = 0.05)

(x_training, y_training), (x_testing, y_testing) = (x_training, y_training), (x_testing, y_testing)

max_words = 500
x_training = sequence.pad_sequences(x_training, maxlen = max_words)
x_testing = sequence.pad_sequences(x_testing, maxlen = max_words)

embedding_size=32
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
			 
batch_size = 64
num_epochs = 3
X_valid, y_valid = x_training[:batch_size], y_training[:batch_size]
X_train2, y_train2 = x_training[batch_size:], y_testing[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])