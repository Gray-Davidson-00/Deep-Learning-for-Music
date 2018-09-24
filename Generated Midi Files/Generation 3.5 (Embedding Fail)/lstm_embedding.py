""" This module prepares midi file data and feeds it to the neural
    network for training """

#! pip install gensim
#! pip install --upgrade gensim

import os.path
import glob
import pickle
import numpy as np

from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from keras.layers import Activation
from keras.layers import Embedding

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.optimizers import RMSprop

from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from gensim.utils import simple_preprocess
from gensim import models


def train_network():
    """ Train a Neural Network to generate music """
    if os.path.isfile('data/notes'):
        notes = pickle.load(open( "data/notes", "rb" ))

    else:
       notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))
    embedding_input, network_input, network_output, note_to_int = prepare_sequences(notes, n_vocab)

    embedding_weights = create_embedding(embedding_input, n_vocab, note_to_int)


    model = create_network(network_input, n_vocab, embedding_weights)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []
    embedding_input = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        embedding_input.append(sequence_in)
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (embedding_input, network_input, network_output, note_to_int)

def create_embedding(embedding_input, n_vocab, note_to_int):
    """
    :Purpose: create an embedding from a set of text data, Save the embedding as a correctly named model
    
    
    :param data_folder: a string which is a path to a folder where text is stored. 
    :type data_folder: string
    
    :param filename: the sub_folder containing a particular author's works. 
    :type filename: string

    :return: returns a gensim embedding layer which represents every distinct word in a corpus as a vector of length "size" 
    :return type: gensim.models.word2vec.Word2Vec
    """
    
    #data_path = data_folder+filename+'/'
    # define training data
    #documents = get_all_text(data_path)
#    sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
#           ['this', 'is', 'the', 'second', 'sentence'],
#           ['yet', 'another', 'sentence'],
#           ['one', 'more', 'sentence'],
#           ['and', 'the', 'final', 'sentence']]
    print("Creating Embedding:")
    if os.path.isfile('music_embedding.bin'):
        embedding = Word2Vec.load('music_embedding.bin')
    else:

        documents = embedding_input
        # train embedding
        bigram_transformer = Phrases(documents)
        bigrams = Phraser(bigram_transformer)
        embedding = Word2Vec(bigrams[documents], sg=1, size=150, window=5, min_count=1, workers=4)
        # sg = 1 gives us skip-gram
        # Size gives us the length of the embedding
        # Window is "The maximum distance between the current and predicted word within a sentence" which I believe is the skip size. 
        # min_count (int) Ignores all words with total frequency lower than this

        # summarize the loaded embedding
        #print(embedding)
        # summarize vocabulary
    #    words = list(embedding.wv.vocab)
    #    print(words)
        # access vector for one word"
    #    print(embedding['river'])
        # save embedding
        embedding.save('music_embedding.bin')
    # load embedding
#    new_model = Word2Vec.load('music_embedding.bin')
#    print(new_model)
# Create a weight matrix for words in training docs
    embedding_weights = np.zeros((n_vocab, 150))
    for note,index in note_to_int.items():
        embedding_weights[index,:] = embedding[note] if note in embedding else np.random.rand(150)
    return embedding_weights

def create_network(network_input, n_vocab, embedding_weights):
    """ create the structure of the neural network """
    print("Creating Network:")
    model = Sequential()
    #model.add inut layer
    model.add(Embedding(input_dim = n_vocab, 
        output_dim=150, 
        input_length = network_input.shape[1], 
        trainable = False,
        weights=[embedding_weights]))

    model.add(CuDNNLSTM(
        256,
        input_shape=(network_input.shape[1], 1)
        # return_sequences=True
    ))
    model.add(Dense(n_vocab))
    #model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    opt = RMSprop(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    # mess with the learning rate in rmsprop
    # add a dense layer after the lstm >> Done
    # remove the dropout >> Done
    # Increase the size of the LSTM >> Done, 128 > 256

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
        mode='auto'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
