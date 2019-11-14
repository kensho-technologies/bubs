# Bubs

Bubs is a Keras/TensorFlow reimplementation of the Flair Contextualized Embeddings (https://alanakbik.github.io/papers/coling2018.pdf). It was developed as a building block for use in Keras/TensorFlow natural language models by [Yuliya Dovzhenko](https://github.com/ydovzhenko) and the [Kensho Technologies AI Research Team](https://www.kensho.com/) ([full contributor list](https://github.com/kensho-technologies/bubs/blob/master/AUTHORS.md)).
 
----------------------------------------------

Bubs implements two types of Flair embeddings:  `news-forward-fast` and `news-backward-fast`.

Bubs consists of two parts:
* ContextualizedEmbedding: a Keras custom layer, which computes the contextualized embeddings. It has two outputs corresponding to the `news-forward-fast` and `news-backward-fast` embeddings.
* InputEncoder: an object for constructing inputs to the ContextualizedEmbedding layer.


#### ContextualizedEmbedding
This layer consists of:
* two character-level embedding layers (one to be used as input to the forward LSTM, one to be used as an input to the backward LSTM)
* two character-level LSTM layers (one going forward, one going backward along the sentence) 
* two indexing layers for selecting character-level LSTM outputs at the locations where tokens end/begin (resulting in two output vectors per token).
* two masking layers to make sure the outputs at padded locations are set to zeros. This is necessary because sentences will have different numbers of tokens and the outputs will be padded to max_token_sequence_length.

The following inputs to the ContextualizedEmbedding layer are required:
* `forward_input`: padded array of character codes corresponding to each sentence with special begin/end characters
* `backward_input`: padded array of character codes in reverse order with special begin/end characters
* `forward_index_input`: padded array of locations of token outputs in forward_input
* `backward_index_input`: padded array of locations of token outputs in backward_input
* `forward_mask_input`: mask of same shape as forward_index_input, with 0's where padded and 1's where real tokens
* `backward_mask_input`:mask of same shape as back_index_input, with 0's where padded and 1's where real tokens

#### InputEncoder

This class provides two methods for preparing inputs to the ContextualizedEmbedding layer:

* `input_batches_from_raw_text()` will accept a raw text string, split it into sentences and tokens, enforce character and token limits by breaking longer sentences into parts. It will then translate characters into numeric codes from the dictionary in `char_to_int.py`, pad sentences to the same length, and compute indices of token-level outputs from the character-level LSTMs.

* `prepare_inputs_from_pretokenized()` will accept a list of lists of tokens and output model inputs . Use at your own risk: this function will not enforce character or token limits and will assume that all sentences fit into one batch. Make sure you split all your sentences into batches before calling this function. Otherwise the indices in `forward_index_input` and `backward_index_input` will be incorrect.

### The model weights

The weights of the ContextualizedEmbedding layer were copied from the corresponding weights inside flair's `news-forward-fast` and `news-backward-fast` embeddings (see `scripts/extracting_model_weights.py` for a code snippet that was used to extract the weights).

### The name Bubs

Bubs is named after the author's cat, Bubs (short for Bubbles).

### A minimal example model
Below we define a very simple example that outputs contextualized embeddings for the following text: "Bubs is a cat. Bubs is cute.".

```python
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from bubs import ContextualizedEmbedding, load_weights_from_npz
from bubs.helpers import InputEncoder


MAX_TOKEN_SEQUENCE_LEN = 125
MAX_CHAR_SEQUENCE_LEN = 2500

"""Load the default weights (provided with this package). If you would like to provide your own 
weights, you may pass a path to the weights npz file to the load_weights_from_npz() function.
"""
weights = load_weights_from_npz()
context_embedding_layer = ContextualizedEmbedding(MAX_TOKEN_SEQUENCE_LEN, weights)

"""Required: define inputs to the ContextualizedEmbedding layer"""
forward_input = Input(shape=(None,), name="forward_input", dtype="int16")
backward_input = Input(shape=(None,), name="backward_input", dtype="int16")
forward_index_input = Input(
    batch_shape=(None, MAX_TOKEN_SEQUENCE_LEN, 2), name="forward_index_input", dtype="int32"
)
forward_mask_input = Input(
    batch_shape=(None, MAX_TOKEN_SEQUENCE_LEN), name="forward_mask_input", dtype="float32"
)
backward_index_input = Input(
    batch_shape=(None, MAX_TOKEN_SEQUENCE_LEN, 2), name="backward_index_input", dtype="int32"
)
backward_mask_input = Input(
    batch_shape=(None, MAX_TOKEN_SEQUENCE_LEN), name="backward_mask_input", dtype="float32"
)

all_inputs = [
    forward_input,
    backward_input,
    forward_index_input,
    backward_index_input,
    forward_mask_input,
    backward_mask_input,
]

forward_embeddings, backward_embeddings = context_embedding_layer(all_inputs)

model = Model(inputs=all_inputs, outputs=[forward_embeddings, backward_embeddings])
model.compile(optimizer=Adam(), loss="categorical_crossentropy")
```
Now, let's get contextualized embeddings for each token in a couple of sentences.
```python
# Initialize an InputEncoder for creating model inputs from raw text sentences
input_encoder = InputEncoder(MAX_TOKEN_SEQUENCE_LEN, MAX_CHAR_SEQUENCE_LEN)

# Embed a couple of test sentences
raw_text = "Bubs is a cat. Bubs is cute."

(
    generator,
    num_batches,
    document_index_batches
) = input_encoder.input_batches_from_raw_text(raw_text, batch_size=128)

# Only one batch, so we use the generator once
forward_embedding, backward_embedding = model.predict_on_batch(next(generator))
```

The shape of each output will be (2, 125, 1024) for:
* 2 sentences
* 125 words in a padded sentence = `MAX_TOKEN_SEQUENCE_LEN`
* 1024: dimension of the embedding for each word
    
Note that the outputs are padded with zeros from the left. For example, to get the forward and backward embedding of the word 'Bubs' in the first sentence, you would need to index the following locations in the model outputs:
`forward_embedding[0, -5]` 
`backward_embedding[0, -5]`

The embeddings for the word 'cat' are: `forward_embedding[0, -2]` and 
`backward_embedding[0, -2]`

### Installation

First, install tensorflow or tensorflow-gpu, depending on your system:

```pip install tensorflow-gpu==1.7.1```
or
```pip install tensorflow==1.7.1```

Tensorflow versions `1.7.1`, `1.10`, and `1.13.1` pass the tests.

Then, install Bubs:

```pip install bubs```

# License

Licensed under the Apache 2.0 License. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Copyright 2019 Kensho Technologies, Inc.
