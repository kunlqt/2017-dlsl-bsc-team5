
'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://qwone.com/~jason/20Newsgroups/
'''

from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.utils.visualize_util import plot
import sys

#Directoris, variables globals.
BASE_DIR = '/home/marta/2017-dlsl-bsc-team5/dlsl-master/news20/'
GLOVE_DIR = BASE_DIR + 'glove.6B/'
TRAIN_TEXT_DATA_DIR = BASE_DIR + '20news-bydate/20news-bydate-train/'
TEST_TEXT_DATA_DIR = BASE_DIR + '20news-bydate/20news-bydate-test/'
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100

BATCH_SIZE=50
OPTIMIZER='rmsprop'
number_of_Dense_layers=2

'''
epoch_to_load=20

 #Eliminar des de declaració embedding fins a fit
with open(BASE_DIR + 'models/model'+ epoch_to_load + '.json','r') as model_json:
    model=model_from_json(model_json.read())

model.load_weights(BASE_DIR + 'models/model_weights_' + epoch_to_load + '_.h5')
'''


def read_glove_vectors(filename):
    embeddings_index = {}
    f = open(filename)
    coefs = None
    for i, line in enumerate(f):
        values = line.split()
        word = values[0]
        if coefs is None:
            coefs = [[0] * len(values[1:])]
        coefs.append(values[1:])
        embeddings_index[word] = i + 1
    f.close()
    coefsm = np.asarray(coefs, dtype='float32')
    return coefsm, embeddings_index


# Converts a list of texts to a matrix of word indices
def test_to_sequence(texts, index, max_sequence_length):
    texts = map(text_to_word_sequence, texts)
    matrix = np.array(pad_sequences([[index[word] for word in text if word in index] for text in texts], max_sequence_length))
    return matrix


# first, build index mapping words to rows the embeddings matrix
print('Reading word vectors.')
embedding_matrix, embeddings_index = read_glove_vectors(os.path.join(GLOVE_DIR, 'glove.6B.%dd.txt' % EMBEDDING_DIM))
print('Found %s word vectors.' % len(embeddings_index))
#print(embedding_matrix.shape)

#exit()

# second, prepare text samples and their labels
print('Processing text dataset')

def prepare_texts(text_data_dir, labels_index = {}):
    texts = []  # list of text samples
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = labels_index.get(name)
            if label_id is None:
                label_id = len(labels_index)
                labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)

    return texts, labels, labels_index

train_texts, train_labels, labels_index = prepare_texts(TRAIN_TEXT_DATA_DIR)
print('Found %s training texts.' % len(train_texts))
test_texts, test_labels, labels_index = prepare_texts(TEST_TEXT_DATA_DIR, labels_index)
print('Found %s test texts.' % len(test_texts))

# finally, vectorize the text samples into a 2D integer tensor
X_train = test_to_sequence(train_texts, embeddings_index, MAX_SEQUENCE_LENGTH)
X_val = test_to_sequence(test_texts, embeddings_index, MAX_SEQUENCE_LENGTH)

y_train = np.array(train_labels, dtype='int32')
y_val = np.array(test_labels, dtype='int32')

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x) #softmax normalizes between 1 and 0.

model = Model(sequence_input, preds)
plot(model, show_shapes=True, to_file='news20.png', show_layer_names=False)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['acc'])
'''
# happy learning!
model.fit(X_train, y_train[..., np.newaxis],
          validation_data=(X_val, y_val[..., np.newaxis]),
          nb_epoch=20, batch_size=128)

'''

tr_loss = []
val_loss=[]
tr_acc=[]
val_acc=[]


# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit(X_train, y_train[..., np.newaxis], batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val[..., np.newaxis])) # add a new dim to y_train and y_val to match output
    tr_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])
    tr_acc.extend(history.history['acc'])
    val_acc.extend(history.history['val_acc'])
    print
    preds = model.predict_classes(X_val, verbose=0)
    save(y_val, preds, 'rnn_{}.pred'.format(iteration))

    make_dir()
    model.save_weights(BASE_DIR + 'models/model_weights_' + iteration + '_.h5')
    with open(BASE_DIR + 'models/model'+ iteration + '.json','w') as model_json:
        model_json.write(model.to_json())

    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[ind], y_val[ind]
        pred = preds[ind]
        q = ctable.decode(rowX)
        correct = ptable.decode(rowy, ch=' ').strip()
        guess = ptable.decode(pred, ch=' ').strip()
        print('W:', q[::-1] if INVERT else q)
        print('T:', correct)
        print(colors.ok + '1' + colors.close if correct == guess else colors.fail + '0' + colors.close,
              guess, '(' + str(levenshtein(correct.split(), guess.split())) + ')')
        print('---')



np.savetxt(BASE_DIR + 'Results/tr_losses_'+ BATCH_SIZE + '_' + OPTIMIZER + '_' + number_of_Dense_layers + '.txt', tr_loss)
np.savetxt(BASE_DIR + 'Results/val_loss_'+ BATCH_SIZE + '_' + OPTIMIZER + '_' + number_of_Dense_layers + '.txt', val_loss)
np.savetxt(BASE_DIR + 'Results/tr_acc_'+ BATCH_SIZE + '_' + OPTIMIZER + '_' + number_of_Dense_layers + '.txt',tr_acc)
np.savetxt(BASE_DIR + 'Results/val_acc_'+ BATCH_SIZE + '_' + OPTIMIZER + '_' + number_of_Dense_layers + '.txt',val_acc)
