from flask import Flask, request,render_template, jsonify
from keras import models
from keras.models import Model
from keras.layers import Input
import numpy as np
#from loadModel import *


#model=models.load_model("/content/drive/My Drive/NLP-Sato/MyModel")

app = Flask(__name__,static_folder='static/css')

#encoder_model,decoder_model=init()

latent_dim=256

input_characters = [' ', '!', '%', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '5', '8', '9', ':',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', '\xa0', '«', '»', 'À', 'Ç', 'É', 'Ê', 'à', 'â', 'ç', 'è', 'é', 'ê', 'î', 'ï', 'ô',
        'ù', 'û', 'œ', '\u2009', '’', '\u202f']

target_characters =['\t', '\n', ' ', '!', '"', '$', '%', '&', "'", ',', '-', '.', '0', '1', '2', '3', '5', '7',
    '8', '9', ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'é']

number_encoder_tokens = len(input_characters)
max_encoder_seq_length = 57
max_decoder_seq_length = 17
number_decoder_tokens= len(target_characters)
encoder_input_data1 = np.zeros(
        (1, max_encoder_seq_length, number_encoder_tokens), dtype='float32'
    )
input_token_index = dict(
        [(char,i) for i, char in enumerate(input_characters)]
    )
target_token_index = dict(
        [(char,i) for i, char in enumerate(target_characters)]
    )
model=models.load_model("./../MyModel")
encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    #lang=request.form['languages']
    #lang=lang.lower()      
    text = decode_sequence(transform(message))
    return jsonify({'prediction':text})
#render_template('index.html', prediction=text)  


def transform(input_text):
    
    for t,char in enumerate(input_text):
        encoder_input_data1[:,t,input_token_index[char]] =1
    encoder_input_data1[:,t+1:,input_token_index[' ']] = 1
    return encoder_input_data1

def decode_sequence(input_seq):

        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, number_decoder_tokens))
        target_seq[0, 0, target_token_index["\t"]] = 1.0

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1, number_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            states_value = [h, c]
        return decoded_sentence

if __name__ == '__main__':
	app.run(debug=True)