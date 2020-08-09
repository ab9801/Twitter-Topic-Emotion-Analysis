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

#model = load_model('lstm-model.h5')
#model.load_weights('lstm-weights.hdf5')
model = load_model('newem_model.h5')
model.load_weights('newem_weights.hdf5')
emotions=[]

data = pd.read_csv("sttweets.csv")
loc_dict={}
for index,row in data.iterrows():
	#print(row['b'],"   ",row['location'])
	y=row['location']
	if ',' in y:
		z=y.split(',')
		y=z[1]
	if y in loc_dict.keys():
		tweets=loc_dict[y]
		tweets.append(row['b'])
		loc_dict[y]=tweets
	else:
		tweet=[]
		tweet.append(row['b'])
		loc_dict[y]=tweet
#print(loc_dict)


e = pd.read_csv("textemotion.csv")
e['target'] = e.sentiment.astype('category').cat.codes
emots = e.sentiment.astype('category')
emot_dict = dict(enumerate(emots.cat.categories))
print(emot_dict)
tk = Tokenizer()
count=0
count_dict={}
for x in emot_dict.values():
	count_dict[x]=0
#print(count_dict)

for loc in loc_dict.keys():
	tlist=[]
	tweets=loc_dict[loc]
	for t in tweets:
		tlist.append(t)
		tk.fit_on_texts(tlist)
		content_seq = tk.texts_to_sequences(tlist)
		content_seq_padded = pad_sequences(content_seq, maxlen=100)
		predicted = model.predict(content_seq_padded)
		y_classes = np.argmax(predicted, axis=1)
		#y_classes = predicted.argmax(axis=-1)
		for i in y_classes:
			count_dict[emot_dict[i]]+=1
	#print(loc,"  ",count_dict)
	if(count<40):
		labels=[]
		number=[]
		count=count+1
		for em in count_dict.keys():
			labels.append(em)
			number.append(count_dict[em])
		ind = np.arange(len(labels))
		plt.bar(ind,number)
		plt.xlabel('emotion')
		plt.ylabel('frequency')
		plt.xticks(ind,labels,fontsize=8,rotation=30)
		plt.title(loc)
		plt.show()
	count_dict={}
	for x in emot_dict.values():
		count_dict[x]=0