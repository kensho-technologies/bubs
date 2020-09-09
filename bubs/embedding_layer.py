# Copyright 2019 Kensho Technologies, LLC.
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Lambda, Layer

from .char_to_int import CHAR_TO_INT


DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "flair_news_fast_weights.npz"
)


def make_lstm_weights_for_keras(
    weight_input_to_hidden, weight_hidden_to_hidden, bias_input_to_hidden, bias_hidden_to_hidden
):
    """Make individual weight matrices extracted from a pytorch LSTM into a keras format.

    The names refer to the variables in the LSTM equation, e.g. see
    http://philipperemy.github.io/images/keras_stateful_lstm_2.png for an example.
    weight_hidden_to_hidden = Whh, bias_input_to_hidden = bih, etc

    Args:
        weight_input_to_hidden: numpy array of dimension (4 x lstm_dim, input_dim)
        weight_hidden_to_hidden: numpy array of dimension (4 x lstm_dim, lstm_dim)
        bias_input_to_hidden: numpy array of dimension (4 x lstm_dim)
        bias_hidden_to_hidden: numpy array of dimension (4 x lstm_dim)

    Returns:
        list of three numpy arrays as expected by 'weights' argument of the keras LSTM layer
    """
    return [
        weight_input_to_hidden.T,
        weight_hidden_to_hidden.T,
        bias_input_to_hidden + bias_hidden_to_hidden,
    ]


def load_weights_from_npz(weights_path=None):
    """Load weights for the ContextualizedEmbedding layer from a numpy npz archive.

    Args:
        weights_path: (optional) path to a npz file. If None, we will use
            flair_news_fast_weights.npz provided with this package.

    Returns:
        a dictionary containing four lists of weights:
            "forward_lstm_weights",
            "backward_lstm_weights",
            "char_embed_weights_forward",
            "char_embed_weights_back"
    """
    if not weights_path:
        weights_path = DEFAULT_WEIGHTS_PATH

    weights_from_npz = np.load(weights_path)
    forward_lstm_weights = make_lstm_weights_for_keras(
        weights_from_npz["weight_input_to_hidden_forward"],
        weights_from_npz["weight_hidden_to_hidden_forward"],
        weights_from_npz["bias_input_to_hidden_forward"],
        weights_from_npz["bias_hidden_to_hidden_forward"],
    )
    backward_lstm_weights = make_lstm_weights_for_keras(
        weights_from_npz["weight_input_to_hidden_back"],
        weights_from_npz["weight_hidden_to_hidden_back"],
        weights_from_npz["bias_input_to_hidden_back"],
        weights_from_npz["bias_hidden_to_hidden_back"],
    )
    char_embed_weights_forward = weights_from_npz["char_embed_weights_forward"]
    char_embed_weights_backward = weights_from_npz["char_embed_weights_back"]

    return {
        "forward_lstm_weights": forward_lstm_weights,
        "backward_lstm_weights": backward_lstm_weights,
        "char_embed_weights_forward": char_embed_weights_forward,
        "char_embed_weights_backward": char_embed_weights_backward,
    }


def batch_indexing(inputs):
    """Index a character-level embedding matrix at token end locations.

    Args:
        inputs: a list of two tensors:
            tensor1: tensor of (batch_size, max_char_seq_len, char_embed_dim*2) of all char-level
                embeddings
            tensor2: tensor of (batch_size, max_token_seq_len, 2) of indices of token ends.
                Something like [[[0, 1], [0, 5]], [[1, 2], [1, 3]], ...]. The last dimension is 2
                because pairs of (sentence_index, token_index)

    Returns:
        A tensor of (batch_size, max_token_seq_len, char_embed_dim*2) of char-level embeddings
            at ends of tokens
    """
    embeddings, indices = inputs
    # this will break on deserialization if we simply import tensorflow
    # we have to use keras.backend.tf instead of tensorflow
    return tf.gather_nd(embeddings, indices)


def multiply(inputs):
    """Multiply a 3d tensor by a 2d tensor along the first two dimensions.

    Args:
        inputs: a list of [tensor1, tensor2], one 3d and one 2d. Both tensors should have the same
            0th and 1st dimensions.

    Returns:
        A 3d tensor, which is tensor1 multiplied by tensor2 along first two dimensions
    """
    x, y = inputs
    # this will break on deserialization if we simply import tensorflow
    # we have to use keras.backend.tf instead of tensorflow
    return tf.einsum("ijk,ij->ijk", x, y)


class ContextualizedEmbedding(Layer):
    def __init__(self, max_token_sequence_len, custom_weights, **kwargs):
        """Initialize custom layer, lstm weights and static character embeddings."""
        super().__init__(**kwargs)

        self.output_dim = 2  # (number of sentences by number of tokens)
        self.max_token_sequence_len = max_token_sequence_len

        # Look up length of the known character vocabulary
        self._char_vocab_len = len(CHAR_TO_INT)

        self._forward_lstm_weights = custom_weights["forward_lstm_weights"]
        self._backward_lstm_weights = custom_weights["backward_lstm_weights"]

        self._char_embeddings_forward = custom_weights["char_embed_weights_forward"]
        self._char_embeddings_backward = custom_weights["char_embed_weights_backward"]

        _, self._char_embedding_dim = self._char_embeddings_forward.shape
        self._char_lstm_dim, _ = self._forward_lstm_weights[1].shape

        # Placeholders for layers
        self.char_embed_forward = None
        self.char_embed_backward = None
        self.forward_char_lstm_layer = None
        self.backward_char_lstm_layer = None
        self.indexing_layer = None
        self.mask_multiply_layer = None

    def build(self, input_shape):
        """Build custom layer."""
        self.char_embed_forward = Embedding(
            self._char_vocab_len,
            self._char_embedding_dim,
            weights=[self._char_embeddings_forward],
            trainable=False,
        )
        self.char_embed_backward = Embedding(
            self._char_vocab_len,
            self._char_embedding_dim,
            weights=[self._char_embeddings_backward],
            trainable=False,
        )

        self.forward_char_lstm_layer = LSTM(
            self._char_lstm_dim,
            use_bias=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            trainable=False,
            input_shape=(None, self._char_embedding_dim),
            return_sequences=True,
            name="forward_lstm",
            weights=self._forward_lstm_weights,
        )
        self.backward_char_lstm_layer = LSTM(
            self._char_lstm_dim,
            use_bias=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            trainable=False,
            input_shape=(None, self._char_embedding_dim),
            return_sequences=True,
            name="backward_lstm",
            weights=self._backward_lstm_weights,
        )

        # Select LSTM outputs at token breaks and make sure the rest is set to zeros
        self.indexing_layer = Lambda(
            batch_indexing, output_shape=tuple((self.max_token_sequence_len, self._char_lstm_dim))
        )
        self.mask_multiply_layer = Lambda(
            multiply, output_shape=tuple((self.max_token_sequence_len, self._char_lstm_dim))
        )
        super().build(input_shape)

    def call(self, inputs):
        """Compute forward and backward character-level contextualized embeddings from inputs."""
        (
            forward_input,
            backward_input,
            forward_index_input,
            backward_index_input,
            forward_mask_input,
            backward_mask_input,
        ) = inputs

        forward_embedded_characters = self.char_embed_forward(forward_input)
        backward_embedded_characters = self.char_embed_backward(backward_input)
        forward_lstm_output = self.forward_char_lstm_layer(forward_embedded_characters)
        backward_lstm_output = self.backward_char_lstm_layer(backward_embedded_characters)

        # Now select outputs at locations where tokens end
        forward_indexed_lstm_output = self.indexing_layer(
            [forward_lstm_output, forward_index_input]
        )
        backward_indexed_lstm_output = self.indexing_layer(
            [backward_lstm_output, backward_index_input]
        )

        # multiply outputs by a mask, which is 1's where real tokens and 0's where padded
        forward_output = self.mask_multiply_layer([forward_indexed_lstm_output, forward_mask_input])
        backward_output = self.mask_multiply_layer(
            [backward_indexed_lstm_output, backward_mask_input]
        )
        return [forward_output, backward_output]

    def compute_output_shape(self, input_shape):
        """Output shape is (batch size) x (number of tokens) x (character lstm dimension)."""
        shape = (input_shape[2][0], input_shape[2][1], self._char_lstm_dim)
        return [shape, shape]

    def get_config(self):
        """Necessary in case we want to serialize a model including this custom layer."""
        base_config = super().get_config()
        base_config["max_token_sequence_len"] = self.max_token_sequence_len
        base_config["custom_weights"] = dict({
            "forward_lstm_weights": self._forward_lstm_weights,
            "backward_lstm_weights": self._backward_lstm_weights,
            "char_embed_weights_forward": self._char_embeddings_forward,
            "char_embed_weights_backward": self._char_embeddings_backward
        })
        return base_config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Turn lists of lists back into numpy arrays upon deserialization."""
        config['custom_weights']['forward_lstm_weights'] = [
            np.array(arr) for arr in config['custom_weights']['forward_lstm_weights']
        ]
        config['custom_weights']['backward_lstm_weights'] = [
            np.array(arr) for arr in config['custom_weights']['backward_lstm_weights']
        ]
        config['custom_weights']['char_embed_weights_forward'] = np.array(
            config['custom_weights']['char_embed_weights_forward']
        )
        config['custom_weights']['char_embed_weights_backward'] = np.array(
            config['custom_weights']['char_embed_weights_backward']
        )
        return super(ContextualizedEmbedding, cls).from_config(config, )
