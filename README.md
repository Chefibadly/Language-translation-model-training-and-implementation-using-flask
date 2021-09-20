# Language Translation

Creating English-to-Spanish **language translation model** using ***neural machine translation*** with ***seq2seq architecture***.  

## Background  

* **Encoder-Decoder LSTM model** having ***seq2seq*** architecture can be used to solve ***many-to-many* sequence problems**, where both inputs and outputs are divided over multiple time-steps. <br>

* The **seq2seq architecture** is a type of *many-to-many sequence modeling*, and is commonly used for a variety of tasks:
  * Text-Summarization
  * Chatbot development
  * Conversational modeling
  * **Neural machine translation**

## Dependencies

* Tensorflow
* Keras
* Numpy 

```
pip install -r requirements.txt
```

## Dataset

Corpus Link: http://www.manythings.org/anki/ ->  fra-eng.zip .zip 

## Architecture

* The **Neural Machine Translation** model is based on **Seq2Seq architecture**, which is an **Encoder-Decoder architecture**.  
* It consists of two layers of **Long Short Term Memory** networks:  
  * *Encoder* LSTM
    * Input = Sentence in the original language
    * Output = Sentence in the translated language, with a *start-of-sentence* token
  * *Decoder* LSTM
    * Input = Sentence in the translated language, with the *start-of-sentence* token
    * Output = Translated sentence, with an *end-of-sentence* token  

## Data preprocessing

* Input does not need to be processed.
* Two copies of the translated sentence is needed to be generated.
  * with *start-of-sentence* token
  * with *end-of-sentence* token

## Tokenization

* Divide input sentences into the corresponding list of characters
* Convert the input characters to a matrix  
* Create the input-token-index dictionary for the input
* Get the number of unique characters in the input
* Get the length of the longest sentence in the input
* Divide output and internediary output sentences into the corresponding list of characters
* Convert the output characters to integers  
* Create the output-token-index dictionary for the output
* Get the number of characters in the output
* Get the length of the longest sentence in the output

###vectorizing the data

```

input_texts = [] #french input
target_texts = [] #english target output
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding = 'utf-8') as f:
  lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines)-1)]:
  target_text,input_text, _ = line.split('\t')
  target_text = '\t' + target_text + '\n'
  input_texts.append(input_text)
  target_texts.append(target_text)
  for char in input_text:
    if char not in input_characters:
      input_characters.add(char)
  for char in target_text:
    if char not in target_characters:
      target_characters.add(char)
```

## Padding

Text sentences can be of varying length. However, the LSTM algorithm expects input instances with the same length. Therefore, we convert our input and output sentences into fixed-length vectors.

* Pad the input sentences upto the length of the longest input sentence
  * zeros are padded at the beginning and words are kept at the end as the encoder output is based on the words occurring at the end of the sentence
* Pad the output sentences upto the length of the longest output sentence  
  * zeros are padded at the end in the case of the decoder as the processing starts from the beginning of a sentence  

## transform encode/decode target and input in matricies zeroes and ones

Deep learning models work with numbers, therefore we need to convert our words into their corresponding numeric vector representations


```
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
  for t, char in enumerate(input_text):
    encoder_input_data[i,t,input_token_index[char]] = 1
  encoder_input_data[i,t + 1:,input_token_index[' ']] = 1
  for t, char in enumerate(target_text):
    decoder_input_data[i,t, target_token_index[char]] = 1
    if t > 0:
      decoder_target_data[i, t - 1, target_token_index[char]] = 1
  decoder_input_data[i, t + 1:, target_token_index[' ']] = 1
  decoder_target_data[i, t:,target_token_index[' ']] = 1

```

## Create the model

* Prepare the encoder and decoder LSTM layers
* Define the decoder output
  * Each character in the output can be any of the total number of unique characters in the output.
  * For each input sentence, we need a corresponding output sentence.  
  * *Hence*, **shape of the output** = (no. of inputs, length of the output sentence, no. of characters in the output)
* Create one-hot encoded output vector for the output Dense layer with 'softmax' as an activation
* Create encoder and decoder
  * Input to the encoder will be the sentence in French
  * Output from the encoder will be the hidden state and cell state of the LSTM (we only need the states of the encoder we don't need the output)
  * Inputs to the decoder will be the the hidden state and cell state from the encoder
  * Output from the decoder will be the sentence with start of sentence tag appended at the beginning
* Define final output layer
  * Dense layer

### Summary of the model

* encoder_inputs and decoder_inputs are passed to the model as inputs, and decoder_outputs as predicted value
* encoder layer is the encoder LSTM.
* There are three outputs from the lstm_1 layer: the output, the hidden layer and the cell state.  
* The cell state and the hidden state are passed to the decoder.
* decoder_lstm contains the output sentences with <sos> token appended at the start, which is embedded and passed through the decoder.
* decoder_lstm layer is the decoder LSTM. 
* The output from the decoder LSTM is passed through the dense layer to make predictions.

## Modifying model for prediction

While training, we know the actual inputs to the decoder for all the output words in the sequence. The input to the decoder and output from the decoder is known and the model is trained on the basis of these inputs and outputs. But, while making actual predictions, the full output sequence is not available. The output word is predicted on the basis of the previous word, which in turn is also predicted in the previous time-step. During prediction the only word available to us is <sos>.

The model is therefore needed to be modified for making predictions so that it follows the following process:

* The encoder model remains the same.  
* In the first step, the sentence in the original language is passed through the encoder, and the hidden and the cell state is the output from the encoder.
* The hidden state and cell state of the encoder, and the <sos>, is used as input to the decoder.
* The decoder predicts a word which may or may not be true.
* In the next step, the decoder hidden state, cell state, and the predicted word from the previous step is used as input to the decoder.
* The decoder predicts the next word.
* The process continues until the <eos> token is encountered.
* All the predicted outputs from the decoder are then concatenated to form the final output sentence.

## Create the prediction model

* The encoder model remains the same.  
* In each step we need the decoder hidden and the cell states.
* In each step there is only single word in decoder input. So, decoder layer needs to be modified.
* The decoder output is defined.
* The decoder output is passed through dense layer to make predictions. 
* basically the prediction model is created using the states from the model's layers
## implement to flask

* download the model and load it to your project
* get the encoder and decoder from the model's layers
* define a transfrom fucntion to transform the input into matrix
* define a decode function to decode the output we get from the decoder

## Improvements

* The model is trained for 100 epochs for hardware constraints. The number of epochs can be modified (increased) to get better results. 
  * Higher number of epochs may give higher accuracy but the model may overfit. 
* To reduce overfitting, we can drop out or add more records to the training set.
* The model is trained to predict short sentences, for longer sentences we cann add the an attention approach

## Conclusion

* **Neural machine translation** is an advance application of Natural Language Processing and involves a very complex architecture.
* We can perform neural machine translation via the ***seq2seq architecture***, which is in turn based on the ***encoder-decoder model***.
* The encoder and decoder are LSTM networks
* The encoder encodes input sentences while the decoder decodes the inputs and generate corresponding outputs.  

# References
* [Fundamentals of LSTM] (https://arxiv.org/pdf/1808.03314.pdf)
* [Attention-based approach] (https://aclanthology.org/W16-2360.pdf)
* [Keras LSTM documentation] (https://keras.io/api/layers/recurrent_layers/lstm/)