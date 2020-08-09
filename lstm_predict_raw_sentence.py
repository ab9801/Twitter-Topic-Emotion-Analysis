import keras 
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
plt.style.use('ggplot')
emotions=[]
#%matplotlib inline
#from IPython.core.display import display, HTML
data = pd.read_csv('textemotion.csv')

print(data.sentiment.value_counts())

data['target'] = data.sentiment.astype('category').cat.codes
#print("target",data['target'])
data['num_words'] = data.content.apply(lambda x : len(x.split()))

bins=[0,50,75, np.inf]
data['bins']=pd.cut(data.num_words, bins=[0,100,300,500,800, np.inf], labels=['0-100', '100-300', '300-500','500-800' ,'>800'])

word_distribution = data.groupby('bins').size().reset_index().rename(columns={0:'counts'})

print(word_distribution.head())

num_class = len(np.unique(data.sentiment.values))
y = data['target'].values


MAX_LENGTH = 500
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.content.values)
content_seq = tokenizer.texts_to_sequences(data.content.values)
content_seq_padded = pad_sequences(content_seq, maxlen=MAX_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(content_seq_padded, y, test_size=0.05)

vocab_size = len(tokenizer.word_index) + 1

'''
inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size,
                            128,
                            input_length=MAX_LENGTH)(inputs)

x = LSTM(64)(embedding_layer)
x = Dense(32, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

print(model.summary())

filepath="lstm-weights.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
          shuffle=True, epochs=3, callbacks=[checkpointer])

model.save('lstm-model.h5')
df = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
sns.pointplot(x="epochs", y="accuracy", data=df, fit_reg=False)
plt.show()
sns.pointplot(x="epochs", y="validation_accuracy", data=df, fit_reg=False, color='green')
plt.show()
'''

model = load_model('lstm-model.h5')
model.load_weights('lstm-weights.hdf5')

'''
#plt.yticks(np.arange(1,5,1))
plt.plot([1,2,3,4,5,6,7,8,9,10],[0.25,0.43,0.59,0.74,0.84,0.86,0.88,0.90,0.905,0.91],marker='o')
plt.xticks(np.arange(1,11,1))
plt.xlabel('epochs (x28500 iterations)')
plt.ylabel('accuracy')
plt.show()

plt.plot([1,2,3,4,5,6,7,8,9,10],[2.09,1.75,1.34,0.98,0.72,0.49,0.38,0.34,0.33,0.32],marker='o')
plt.xticks(np.arange(1,11,1))
plt.xlabel('epochs (x28500 iterations)')
plt.ylabel('Loss')
plt.show()
'''

emots = data.sentiment.astype('category')
emot_dict = dict(enumerate(emots.cat.categories))

text = ["I am enjoying today. Best day of my life. Very happy.","Very bored today Have nothing to do. jobless.","How did this happen? Shocked to say the least. Cannot believe it.","no one talk to me. just want to be alone right now."]
#text = ["happily enjoying my day.","no one talk to me. just want to be alone right now.","That's it. goodbye. I don't like you."]
t=0

for t in range(len(text)):
	print("Text",t+1,":",text[t])
tk = Tokenizer()
#tk.fit_on_texts(text)
tokenizer.fit_on_texts(text)
cs = tokenizer.texts_to_sequences(text)
csq = pad_sequences(cs, maxlen=MAX_LENGTH)
print("\n\n")
print("Emotions:\n")
predicted = model.predict(csq)
#print(predicted)
#y_classes = np.argmax(predicted, axis=1)
y_classes = predicted.argmax(axis=-1)
#print(y_classes)
for i in y_classes:
	emotions.append(emot_dict[i])
for t in range(len(emotions)):
	print("Text",t+1,":",emotions[t])
print("\n\n")