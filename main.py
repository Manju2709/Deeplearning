import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read)
import collections
ques = pd.read_csv('Questions.csv',encoding='iso-8859-1')
tags = pd.read_csv('Tags.csv',encoding='iso-8859-1')
tagCount =  collections.Counter(list(tags['Tag'])).most_common(10)
top10=['javascript','java','c#','php','android','jquery','python','html','c++','ios']
tag_top10= tags[tags.Tag.isin(top10)]
print (tag_top10.shape)
print ("Starting")
def add_tags(question_id):
    return tag_top10[tag_top10['Id'] == question_id['Id']].Tag.values
top10 = tag_top10.apply(add_tags, axis=1)
tag_top10=pd.concat([tag_top10, top10.rename('Tags')], axis=1)
tag_top10.drop(["Tag"], axis=1, inplace=True)
top10_tags=tag_top10.loc[tag_top10.astype(str).drop_duplicates().index]
ques = pd.read_csv('Questions.csv',encoding='iso-8859-1')
total=pd.merge(ques, top10_tags, on='Id')
print("Labeling over")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, BatchNormalization, GRU ,concatenate
from keras.models import Model
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(total.Tags)
labels = multilabel_binarizer.classes_
labels
train,test=train_test_split(total[:550000],test_size=0.25,random_state=24)
X_train_t=train['Title']
X_train_b=train['Body']
y_train=multilabel_binarizer.transform(train['Tags'])
X_test_t=test['Title']
X_test_b=test['Body']
y_test=multilabel_binarizer.transform(test['Tags'])
sent_lens_t=[]
print("Setting up training")
import nltk
nltk.download('punkt')
for sent in train['Title']:
    sent_lens_t.append(len(word_tokenize(sent)))
max_len_t = 18
tok = Tokenizer(char_level=False,split=' ')
tok.fit_on_texts(X_train_t)
sequences_train_t = tok.texts_to_sequences(X_train_t)
vocab_len_t=len(tok.index_word.keys())
vocab_len_t=len(tok.index_word.keys())
sequences_matrix_train_t = sequence.pad_sequences(sequences_train_t,maxlen=max_len_t)
sequences_test_t = tok.texts_to_sequences(X_test_t)
sequences_matrix_test_t = sequence.pad_sequences(sequences_test_t,maxlen=max_len_t)
sent_lens_b=[]
print("Seting body")
for sent in train['Body']:
    sent_lens_b.append(len(word_tokenize(sent)))
max(sent_lens_b)
np.quantile(sent_lens_b,0.90)
max_len_b = 600
tok = Tokenizer(char_level=False,split=' ')
tok.fit_on_texts(X_train_b)
sequences_train_b = tok.texts_to_sequences(X_train_b)
vocab_len_b =len(tok.index_word.keys())
sequences_matrix_train_b = sequence.pad_sequences(sequences_train_b,maxlen=max_len_b)
sequences_test_b = tok.texts_to_sequences(X_test_b)
sequences_matrix_test_b = sequence.pad_sequences(sequences_test_b,maxlen=max_len_b)
sequences_matrix_train_t.shape,sequences_matrix_train_b.shape,y_train.shape
sequences_matrix_test_t.shape,sequences_matrix_test_b.shape,y_test.shape

print("RNN Training Starting")


def RNN():
    # Title Only
    title_input = Input(name='title_input', shape=[max_len_t])
    title_Embed = Embedding(vocab_len_t + 1, 2000, input_length=max_len_t, mask_zero=True, name='title_Embed')(
        title_input)
    gru_out_t = GRU(300)(title_Embed)
    # auxiliary output to tune GRU weights smoothly
    auxiliary_output = Dense(10, activation='sigmoid', name='aux_output')(gru_out_t)

    # Body Only
    body_input = Input(name='body_input', shape=[max_len_b])
    body_Embed = Embedding(vocab_len_b + 1, 170, input_length=max_len_b, mask_zero=True, name='body_Embed')(body_input)
    gru_out_b = GRU(200)(body_Embed)

    # combined with GRU output
    com = concatenate([gru_out_t, gru_out_b])

    # now the combined data is being fed to dense layers
    dense1 = Dense(400, activation='relu')(com)
    dp1 = Dropout(0.5)(dense1)
    bn = BatchNormalization()(dp1)
    dense2 = Dense(150, activation='relu')(bn)

    main_output = Dense(10, activation='sigmoid', name='main_output')(dense2)

    model = Model(inputs=[title_input, body_input], outputs=[main_output, auxiliary_output])
    return model
model = RNN()
model.summary()
model.compile(optimizer='adam',loss={'main_output': 'categorical_crossentropy', 'aux_output': 'categorical_crossentropy'},
              metrics=['accuracy'])
results=model.fit({'title_input': sequences_matrix_train_t, 'body_input': sequences_matrix_train_b},
          {'main_output': y_train, 'aux_output': y_train},
          validation_data=[{'title_input': sequences_matrix_test_t, 'body_input': sequences_matrix_test_b},
          {'main_output': y_test, 'aux_output': y_test}],
          epochs=5, batch_size=800)
(predicted_main, predicted_aux)=model.predict({'title_input': sequences_matrix_test_t, 'body_input': sequences_matrix_test_b},verbose=1)
from sklearn.metrics import classification_report,f1_score
print(f1_score(y_test,predicted_main>.55,average='samples'))
print(classification_report(y_test,predicted_main>.55))