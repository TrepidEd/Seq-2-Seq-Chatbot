from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np

#Variables
training_batch_size = 64  # Batch size for training
batch_size = 20
loop_count = 5
epochs = 200  # Number of epochs to train for
latent_dim = 256  # Latent dimensionality of the encoding space
num_samples = 10000
enc_path = './Data/train.enc'
dec_path = './Data/train.dec'

# Vectorize the data
input_texts = []
target_texts = []
input_vocab = set()
target_vocab = set()


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

"""Read encoder-decoder input data"""
enc_lines = open(enc_path)
for line in enc_lines:
    input_text = intersperse(line.split(), " ")
    input_texts.append(input_text)
    for word in input_text:
        if word not in input_vocab:
            input_vocab.add(word)

dec_lines = open(dec_path)
for line in dec_lines:
    target_text = intersperse(line.split(), " ")
    target_text = ['\t'] + target_text + ['\n']
    target_texts.append(target_text)
    for word in target_text:
        if word not in target_vocab:
            target_vocab.add(word)

#Define network variables
input_vocab = sorted(list(input_vocab))
target_vocab = sorted(list(target_vocab))
num_encoder_tokens = len(input_vocab)
num_decoder_tokens = len(target_vocab)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict([(word, i) for i, word in enumerate(input_vocab)])
target_token_index = dict([(word, i) for i, word in enumerate(target_vocab)])

print("Making Model...")
#Define Encoder sequence
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

#Define Decoder sequence
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
print("Making Complete!")


print("Compiling Model...")
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
print("Compile Complete!")


print("Training Model...")
#Train model
for _ in range(loop_count):
    for index in range(int(len(input_texts)/batch_size)):
        batch_input =   input_texts     [   index   *  batch_size     :   (index+1)     *   batch_size  ]
        batch_target =  target_texts    [   index   *  batch_size     :   (index+1)     *   batch_size  ]
        encoder_input_data =    np.zeros( ( batch_size, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        decoder_input_data =    np.zeros( ( batch_size, max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        decoder_target_data =   np.zeros( ( batch_size, max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        for i, (input_text, target_text) in enumerate(zip(batch_input, batch_target)):
            print(input_text,target_text)
            for t, word in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[word]] = 1.
            print(encoder_input_data[i])
            for t, word in enumerate(target_text):
                decoder_input_data[i, t, target_token_index[word]] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[word]] = 1.
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=epochs, validation_split=0.1)

print("Training Complete!")
print("Saving Model...")
#Save model
model.save('Seq2Seq.h5')
print("Model Saved!")

encoder_model = Model(encoder_inputs, encoder_states)

#For inference.
decoder_state_input_h               =     Input(    shape=(latent_dim,) )
decoder_state_input_c               =     Input(    shape=(latent_dim,) )
decoder_states_inputs               =     [ decoder_state_input_h, decoder_state_input_c    ]
decoder_outputs, state_h, state_c   =     decoder_lstm( decoder_inputs, initial_state=decoder_states_inputs )
decoder_states                      =     [ state_h, state_c    ]
decoder_outputs                     =     decoder_dense(decoder_outputs)
decoder_model                       =     Model(    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states    )

reverse_input_word_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_word_index = dict((i, word) for word, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(    [target_seq] + states_value )

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_token_index]
        decoded_sentence += sampled_word

        # Exit condition: either hit max length or find stop character.
        if (sampled_word == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


testing = True
while testing:
    print('------------NEW LINE------------')
    human_input = input()
    print('Input sentence:', human_input) 
    human_input = intersperse(human_input.split(), " ")
    human_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, word in enumerate(human_input):
        if word in input_token_index.keys():
            human_input_data[0, t, input_token_index[word]] = 1.
    decoded_sentence = decode_sequence(human_input_data)
    print('Decoded sentence:', decoded_sentence)

