import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

df = pd.read_excel('literature_articles_all.xlsx', sheet_name = 'All Evidence_no_aggregation', usecols=['Target Concept', 'Evidence'])
target_concept = list(df["Target Concept"])
evidence = list(df["Evidence"])
 
#distrbtn = df['Target Concept'].value_counts() #class data distribution as textual

fig = plt.figure(figsize=(8,8))
pl = df.groupby('Target Concept').Evidence.count().plot.bar(ylim=0)
plt.show()

"""
Data pre-processing phase with regular expression
"""
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
X = []
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
    X.append(x)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(target_concept)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded) 

"""
Limitation of long texts with max_words variable and turning the training package into numerical status.
"""
max_words = 3500
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
print(len(tok.word_index))
sequences = tok.texts_to_sequences(X)
sequences_matrix = sequence.pad_sequences(sequences)

X_train,X_test,y_train,y_test = train_test_split(sequences_matrix, y, test_size=0.15)

model = Sequential()
model.add(Embedding(max_words,192,input_length=sequences_matrix.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(212, dropout = 0.4, recurrent_dropout = 0.4))
model.add(Dense(28, activation = "softmax"))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer="adam", metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.15)

loss,accuracy = model.evaluate(X_test, y_test)
print("loss: ", loss)
print("accuracy: ", accuracy)

predx = model.predict(X_test)
print(predx)

"""
Single sentence prediciton and label finding phase.
"""
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
    sent_temiz.append(x.strip())

s = tok.texts_to_sequences(sent_temiz)
s = sequence.pad_sequences(s,maxlen = X_train.shape[1])

labels = list(sorted(set(target_concept)))

pred = model.predict(s)
class_ = pred.argmax(axis=-1)
predicted_label = labels[class_[0]]
print(predicted_label)
