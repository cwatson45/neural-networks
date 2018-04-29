# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend
import nn_parser as p
import csv
import random
import numpy as np
np.random.seed(1337)
import json
import re
#import nltk
import string
from keras import backend as K
from keras.models import Sequential
#from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, InputSpec, TimeDistributed, BatchNormalization, Bidirectional, Wrapper, Concatenate, concatenate
from keras.layers import Flatten, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras import regularizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit




def make_safe(x):
    return K.clip(x, K.common._EPSILON, 1.0 - K.common._EPSILON)

class ProbabilityTensor(Wrapper):
    """ function for turning 3d tensor to 2d probability matrix, which is the set of a_i's """
    def __init__(self, dense_function=None, *args, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        #layer = TimeDistributed(dense_function) or TimeDistributed(Dense(1, name='ptensor_func'))
        layer = TimeDistributed(Dense(1, name='ptensor_func'))
        super(ProbabilityTensor, self).__init__(layer, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ProbabilityTensor, self).build()

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,n 
        #       s.t. \sum_n n = 1
        if isinstance(input_shape, (list,tuple)) and not isinstance(input_shape[0], int):
            input_shape = input_shape[0]

        return (input_shape[0], input_shape[1])

    def squash_mask(self, mask):
        if K.ndim(mask) == 2:
            return mask
        elif K.ndim(mask) == 3:
            return K.any(mask, axis=-1)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return self.squash_mask(mask)

    def call(self, x, mask=None):
        energy = K.squeeze(self.layer(x), 2)
        p_matrix = K.softmax(energy)
        if mask is not None:
            mask = self.squash_mask(mask)
            p_matrix = make_safe(p_matrix * mask)
            p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True))*mask
        return p_matrix

    def get_config(self):
        config = {}
        base_config = super(ProbabilityTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SoftAttentionConcat(ProbabilityTensor):
    '''This will create the context vector and then concatenate it with the last output of the LSTM'''
    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,f where f is weighted features summed across n
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return(input_shape[0], 2048)

    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim==2:
            return None
        else:
            raise Exception("Unexpected situation")
    
    def call(self, x, mask=None):
        # b,n,f -> b,f via b,n broadcasted
        p_vectors = K.expand_dims(super(SoftAttentionConcat, self).call(x, mask), 2)
        expanded_p = K.repeat_elements(p_vectors, K.int_shape(x)[2], axis=2)
        context = K.sum(expanded_p * x, axis=1)
        last_out = x[:, -1, :]
        print('\n\n\n***\n')
        print(x.shape)
        print(last_out.shape)
        print(K.concatenate([context, last_out]).shape)
        return K.concatenate([context, last_out]) 


  
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', required=False, type=int, default=10)
parser.add_argument('--batch_size', required=False, type=int, default=128)
parser.add_argument('--numCV', required=False, type=int, default=1)
parser.add_argument('--reg', required=False, type=float, default=.001)
parser.add_argument('--lr', required=False, type=float, default=1e-3)
args = parser.parse_args()
batch_size = args.batch_size
num_epochs = args.num_epochs
numCV = args.numCV
reg = args.reg
lr = args.lr

# 1. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 20

# 2. Classifier hyperparameters
max_sentence_len = 280
min_sentence_length = 7
rankK = 10
#reg = .001
#batch_size = 128

# ========================================================================================
# Preprocess the open bugs, extract the vocabulary and learn the word2vec representation
# ========================================================================================
all_data, all_owner, min_sentence_length, max_sentence_len = p.parse_metadata_bugs_analysts(
    'bugs-2018-02-09.csv')
SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

import numpy_indexed as npi
samples_mask = npi.multiplicity(all_owner) >= 5
all_data = all_data[samples_mask]
all_owner = all_owner[samples_mask]

# Learn the word2vec model and extract vocabulary
wordvec_model = Word2Vec(
    all_data,
    min_count=min_word_frequency_word2vec,
    size=embed_size_word2vec,
    window=context_window_word2vec)
vocabulary = wordvec_model.wv.vocab
vocab_size = len(vocabulary)
print(vocabulary)
print(vocab_size)
totalLength = len(all_data)
splitLength = int(totalLength * 0.8)



# ========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
# ========================================================================================
totalLength = len(all_data)
splitLength = int(totalLength * 0.8)

i = 0
for train_index, test_index in SSS.split(all_data, all_owner):
    i += 1
    print(type(all_data))

    # Split cross validation set

    train_data = all_data[train_index]
    test_data = all_data[test_index]
    train_owner = all_owner[train_index]
    test_owner = all_owner[test_index]

    # Remove words outside the vocabulary
    updated_train_data = []
    updated_train_data_length = []
    updated_train_owner = []
    final_test_data = []
    final_test_owner = []
    for j, item in enumerate(train_data):
        current_train_filter = [word for word in item if word in vocabulary]
        if len(current_train_filter) >= min_sentence_length:
            updated_train_data.append(current_train_filter)
            updated_train_owner.append(train_owner[j])

    for j, item in enumerate(test_data):
        current_test_filter = [word for word in item if word in vocabulary]
        if len(current_test_filter) >= min_sentence_length:
            final_test_data.append(current_test_filter)
            final_test_owner.append(test_owner[j])

    # Remove data from test set that is not there in train set
    train_owner_unique = set(updated_train_owner)
    test_owner_unique = set(final_test_owner)
    unwanted_owner = list(test_owner_unique - train_owner_unique)
    updated_test_data = []
    updated_test_owner = []
    updated_test_data_length = []
    for j in range(len(final_test_owner)):
        if final_test_owner[j] not in unwanted_owner:
            updated_test_data.append(final_test_data[j])
            updated_test_owner.append(final_test_owner[j])

    unique_train_label = list(set(updated_train_owner))
    classes = np.array(unique_train_label)

    # Create train and test data for deep learning + softmax
    X_train = np.empty(
        shape=[
            len(updated_train_data),
            max_sentence_len,
            embed_size_word2vec],
        dtype='float32')
    Y_train = np.empty(shape=[len(updated_train_owner), 1], dtype='int32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence,
    # word indices start with 3
    for j, curr_row in enumerate(updated_train_data):
        sequence_cnt = 0
        for item in curr_row:
            if item in vocabulary:
                X_train[j, sequence_cnt, :] = wordvec_model[item]
                sequence_cnt = sequence_cnt + 1
                if sequence_cnt == max_sentence_len - 1:
                    break
        for k in range(sequence_cnt, max_sentence_len):
            X_train[j, k, :] = np.zeros((1, embed_size_word2vec))
        Y_train[j, 0] = unique_train_label.index(updated_train_owner[j])

    X_test = np.empty(
        shape=[
            len(updated_test_data),
            max_sentence_len,
            embed_size_word2vec],
        dtype='float32')
    Y_test = np.empty(shape=[len(updated_test_owner), 1], dtype='int32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence,
    # word indices start with 3
    for j, curr_row in enumerate(updated_test_data):
        sequence_cnt = 0
        for item in curr_row:
            if item in vocabulary:
                X_test[j, sequence_cnt, :] = wordvec_model[item]
                sequence_cnt = sequence_cnt + 1
                if sequence_cnt == max_sentence_len - 1:
                    break
        for k in range(sequence_cnt, max_sentence_len):
            X_test[j, k, :] = np.zeros((1, embed_size_word2vec))
        Y_test[j, 0] = unique_train_label.index(updated_test_owner[j])

    y_train = np_utils.to_categorical(Y_train, len(unique_train_label))
    y_test = np_utils.to_categorical(Y_test, len(unique_train_label))

print('i=', i)
print(X_train.shape, X_test.shape)


#X_train = np.expand_dims(X_train, axis = 3)
#X_test = np.expand_dims(X_test, axis = 3)


input_shape = X_train.shape[1:] #max sentence length
model = Sequential()
model.add(Bidirectional(LSTM(1024, return_sequences=True, recurrent_dropout=0.5, activity_regularizer = regularizers.l2(reg), input_shape =input_shape), input_shape =input_shape))
model.add(Bidirectional(LSTM(1024, return_sequences=True, recurrent_dropout=0.5, activity_regularizer = regularizers.l2(reg))))
#model.add(Bidiecctional(LSTM(1024, return_sequences=True, recurrent_dropout=0.5, activity_regularizer = regularizers.l2(reg))))
#model.add(Bidiecctional(LSTM(1024, return_sequences=True, recurrent_dropout=0.5, activity_regularizer = regularizers.l2(reg))))
#model.add(Bidiecctional(LSTM(1024, return_sequences=True, recurrent_dropout=0.5, activity_regularizer = regularizers.l2(reg))))
model.add(Dense(1024, activation='relu', activity_regularizer = regularizers.l2(reg)))
model.add(Dropout(rate = .5))
model.add(Flatten())
model.add(Dense(len(unique_train_label), activation='softmax'))
'''
sequence_embed = Input(shape = (max_sentence_len, embed_size_word2vec,))

forwards_1 = LSTM(1024, return_sequences=True, recurrent_dropout=0.5, activity_regularizer = regularizers.l2(reg))(sequence_embed)
attention_1 = SoftAttentionConcat()(forwards_1)
#after_dp_forward_5 = BatchNormalization()(attention_1)

backwards_1 = LSTM(1024, return_sequences=True, recurrent_dropout=0.5, go_backwards=True, activity_regularizer = regularizers.l2(reg))(sequence_embed)
attention_2 = SoftAttentionConcat()(backwards_1)
#after_dp_backward_5 = BatchNormalization()(attention_2)
merged = merge([attention_1, attention_2], mode='concat', concat_axis=-1)

forwards_2 = LSTM(1024, return_sequences=True, recurrent_dropout=0.5, activity_regularizer = regularizers.l2(reg))(merged)
attention_2 = SoftAttentionConcat()(forwards_2)
backwards_3 = LSTM(1024, return_sequences=True, recurrent_dropout=0.5, go_backwards=True, activity_regularizer = regularizers.l2(reg))(merged)
attention_4 = SoftAttentionConcat()(backwards_2)
merged2 = merge([attention_3, attention_4], mode='concat', concat_axis=-1)

after_merge = Dense(1000, input_dim=(4092,), activation='relu', activity_regularizer = regularizers.l2(reg))(merged2)
after_dp = Dropout(0.4)(after_merge)
output = Dense(len(unique_train_label), activation='softmax')(after_dp)

model = Model(inputs=sequence_embed, outputs=output)
'''
#compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
print("\n\nmodel\n\n")
print(model.summary())
print('\n\n')

print('model.layers')
print(model.layers)
print('model.input')
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)

print("Training the Model")
print('reg =', reg)
print('lr = ', lr)
#fit the model
hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split = 0.2)              

predict = model.predict(X_test)        
accuracy = []
sortedIndices = []
pred_classes = []

for ll in predict:
    sortedIndices.append(
        sorted(
            range(
                len(ll)),
            key=lambda ii: ll[ii],
            reverse=True))
thefile = open('indices.txt', 'w')
for item in sortedIndices:
    thefile.write("%s\n" % item)
for k in range(1, rankK + 1):
    id = 0
    trueNum = 0
    for sortedInd in sortedIndices:
        pred_classes.append(classes[sortedInd[:k]])
        if Y_test[id] in sortedInd[:k]:
            trueNum += 1
        id += 1
    accuracy.append((float(trueNum) / len(predict)) * 100)
print('Test accuracy:', accuracy)

train_result = hist.history
print(train_result)
del model
