"""
This module prepares midi file data and feeds it to the neural network for training.


The file takes as its input a filepath where midi files are stored and outputs a file containing the notes and a set of
.hdf5 files containing weights for each epoch of the neural network.

This file is currently constructed to run only on a CUDA-enabled graphics processing unit (GPU).

"""


import glob
import pickle
import numpy
import os

from music21 import converter
from music21 import instrument
from music21 import note
from music21 import chord

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNGRU
from keras.layers import Activation
from keras.layers import Bidirectional

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop


def train_network():
    """
    A compilation method which will call to all the other methods in this .py file.  Its purpose is to read midi files
    and train a  neural network to write music which resembles their contents.

    """

    # If the midi files have already been read in and parsed into a continuous
    # sequence of notes then skip this tedious step.

    if os.path.isfile('data/notes'):
        notes = pickle.load(open( "data/notes", "rb" ))

    else:
        notes = get_notes()

    # Get the total number of pitch names.  This will be used repeatedly, including as the shape-input of various
    # network layers.

    n_vocab = len(set(notes))

    # Create the set of sequences and their corresponding outputs.
    # These will be the inputs and outputs used to train the network.

    network_input, network_output = prepare_sequences(notes, n_vocab)

    # Establish the shape of the network.

    model = create_network(network_input, n_vocab)

    # Train the network using the model, inputs and outputs defined above.  This step will save weights at each epoch.

    train(model, network_input, network_output)


def get_notes():
    """
    Get all the notes and chords from the midi files in the stated directory.  Extract melodic sequences from these and
    concatenate the sequences into a set of all notes in all songs, then save that list to a file.

    return:: all the notes from all songs as a single sequence
    rtype:: list

    """

    notes = []

    for file in glob.glob("full_midi_files/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        # If the file has instrument parts

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()

        # Otherwise the file has notes in a flat structure

        except:
            notes_to_parse = midi.flat.notes

        # Append each note in notes_to_parse to the notes list

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            # If a note is a rest, append it as a rest.

            elif isinstance(element, note.Rest):
                notes.append('Rest')
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        # If the song ends, append a rest.  Later I should figure out a better way of adding an "end of sequence" token.

        notes.append('Rest')

    # Save the notes as a file for later use.

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    # Also return them for immediate use.

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 50

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with 'LSTM' layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(CuDNNGRU(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2])
    ))


    # model.add(CuDNNLSTM(
    #     128,
    #     input_shape=(network_input.shape[1], network_input.shape[2])
    #     # return_sequences=True
    # ))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    opt = RMSprop(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

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
