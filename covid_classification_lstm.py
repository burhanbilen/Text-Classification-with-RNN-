import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

df = pd.read_excel('literature_articles_all.xlsx', sheet_name = 'All Evidence_no_aggregation', usecols=['Target Concept', 'Evidence'])
target_concept = list(df["Target Concept"]) # Getting the input and output columns; evidence:input, target_concept:output
evidence = list(df["Evidence"])
 
#distrbtn = df['Target Concept'].value_counts() #class data distribution as textual

fig = plt.figure(figsize=(8,8))
pl = df.groupby('Target Concept').Evidence.count().plot.bar(ylim=0)
plt.show() # plotting class data distribution as a graphic

"""
Data pre-processing phase with regular expression
"""
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
evidence_clean = []
for sen in range(0, len(evidence)):
    x = re.sub(r'\W', ' ', str(evidence[sen]))
    x = re.sub(r'\s+[a-zA-Z]\s+', ' ', x)
    x = re.sub(r'\^[a-zA-Z]\s+', ' ', x) 
    x = re.sub(r'\s+', ' ', x, flags=re.I)
    x = re.sub(r'^b\s+', '', x)
    x = x.lower()
    x = x.split()
    x = [stemmer.stem(word) for word in x]
    x = ' '.join(x)
    evidence_clean.append(x)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(target_concept) # Enumarating words with LabelEncoder method to turn words into integers.

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded) # One-Hot Encoding on integer valued class labels, for categorical process. 

X_train,X_test,y_train,y_test = train_test_split(evidence_temiz, y, test_size=0.13) # Splitting the data into train and test packages.

"""
Limitation of long texts with max_words variable and turning the training package into numerical status.
"""
max_words = 2000
max_len = 200
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train) # Turning the tokenized train package into sequences.
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len) # making ever single sequence in same length, with max_len variable.  

model = Sequential()
model.add(Embedding(max_words,40,input_length=max_len)) # To embed the integer values and making the sequences have proper dimension to feed LSTM layer.
model.add(LSTM(128)) # RNN LSTM Network, to predict the next value thanks to sequences.
model.add(Dropout(0.5)) # To prevent the over-fitting problem, Dropout was added.
model.add(Dense(96)) # Next, an ANN Dense to increase network size.
model.add(Activation('relu')) 
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(28)) # output (y) sequence length
model.add(Activation('softmax')) # for multiple output and multiple class works softmax function is needed.  

model.summary() # To see the each layer steps summary() function was used.
model.compile(loss='categorical_crossentropy',optimizer="adam", metrics=['accuracy']) # Specifying the loss and optimizer functions for training phase and definition of metrics.
model.fit(sequences_matrix, y_train, epochs=15, batch_size=64, validation_split=0.13) # Model training and test phase. epochs:number of usage of whole dataset, batch_size: number of training data in 1 epochs.

"""
Single sentence prediciton and label finding phase.
"""
# Cleaning the input sentence.
sent = ["In addition to moisture , there was also a significant negative correlation between ether content and TGEV survival ."]
sent_temiz = []
for sen in range(0, len(sent)):
    x = re.sub(r'\W', ' ', str(sent[sen]))
    x = re.sub(r'\s+[a-zA-Z]\s+', ' ', x)
    x = re.sub(r'\^[a-zA-Z]\s+', ' ', x) 
    x = re.sub(r'\s+', ' ', x, flags=re.I)
    x = re.sub(r'^b\s+', '', x)
    x = x.lower()
    x = x.split()
    x = [stemmer.stem(word) for word in x]
    x = ' '.join(x)
    sent_temiz.append(x)

# turning the single input into a sequence and padding its length.
tok.fit_on_texts(sent_temiz)
s = tok.texts_to_sequences(sent_temiz)
s = sequence.pad_sequences(s,maxlen=max_len)

labels = list(sorted(set(target_concept))) # Sorting all classes and making them in a set to prevent repetition.

pred = model.predict(s) # Model prediction output.
class = pred.argmax(axis=-1) # Getting the index of the biggest value in the pred variable.
predicted_label = labels[class[0]] # Label selection with the obtained index.
print(predicted_label)
