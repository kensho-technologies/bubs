# Copyright 2019 Kensho Technologies, LLC.
import os
import unittest

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ..embedding_layer import (
    ContextualizedEmbedding, load_weights_from_npz, make_lstm_weights_for_keras
)
from ..helpers import InputEncoder


class TestEmbeddingLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.max_token_seq_len = 5
        cls.max_char_seq_len = 20
        cls.input_encoder = InputEncoder(cls.max_token_seq_len, cls.max_char_seq_len)

        cur_path = os.path.dirname(os.path.abspath(__file__))
        weights_filename = "dummy_weights.npz"
        weights_path = os.path.join(cur_path, weights_filename)
        weights = load_weights_from_npz(weights_path)

        cls.context_embedding_layer = ContextualizedEmbedding(cls.max_token_seq_len, weights)

        # Define a dummy model with just inputs and ContextualizedEmbedding Layer
        forward_input = Input(shape=(None,), name="forward_input", dtype="int16")
        backward_input = Input(shape=(None,), name="backward_input", dtype="int16")
        forward_index_input = Input(
            batch_shape=(None, cls.max_token_seq_len, 2), name="forward_index_input", dtype="int32"
        )
        forward_mask_input = Input(
            batch_shape=(None, cls.max_token_seq_len), name="forward_mask_input", dtype="float32"
        )
        backward_index_input = Input(
            batch_shape=(None, cls.max_token_seq_len, 2), name="backward_index_input", dtype="int32"
        )
        backward_mask_input = Input(
            batch_shape=(None, cls.max_token_seq_len), name="backward_mask_input", dtype="float32"
        )

        all_inputs = [
            forward_input,
            backward_input,
            forward_index_input,
            backward_index_input,
            forward_mask_input,
            backward_mask_input,
        ]

        forward_embedded_characters, backward_embedded_characters = cls.context_embedding_layer(
            all_inputs
        )

        cls.model = Model(
            inputs=all_inputs, outputs=[forward_embedded_characters, backward_embedded_characters]
        )
        cls.model.compile(optimizer=Adam(), loss="categorical_crossentropy")

    def test_default_weights(self):
        # check that default weights get loaded ok
        weights = load_weights_from_npz()
        context_embedding_layer = ContextualizedEmbedding(self.max_token_seq_len, weights)  # noqa

    def test_custom_layer(self):
        """Build a dummy Keras model using the custom layer. Check the output dimensions."""
        raw_text = "This is a sentence. This is a second sentence."
        batch_size = 2
        gen, num_batches, document_index_batches = self.input_encoder.input_batches_from_raw_text(
            raw_text, batch_size=batch_size
        )
        for batch_idx in range(num_batches):
            forward_embedding, backward_embedding = self.model.predict_on_batch(next(gen))
            expected_embedding_shape = (
                len(document_index_batches[batch_idx]),
                self.max_token_seq_len,
                self.context_embedding_layer._char_lstm_dim,
            )
            self.assertTupleEqual(expected_embedding_shape, forward_embedding.shape)
            self.assertTupleEqual(expected_embedding_shape, backward_embedding.shape)

    def test_make_lstm_weights_for_keras(self):
        """Define a few asymmetric matrices, distinct from each other, as dummy weights."""
        char_lstm_dim = 10
        char_embedding_dim = 5
        weight_input_to_hidden = np.tri(4 * char_lstm_dim, char_embedding_dim)
        weight_hidden_to_hidden = np.tri(4 * char_lstm_dim, char_lstm_dim) * 2
        bias_input_to_hidden = np.ones(4 * char_lstm_dim)
        bias_hidden_to_hidden = np.ones(4 * char_lstm_dim) * 2

        keras_lstm_weights = make_lstm_weights_for_keras(
            weight_input_to_hidden,
            weight_hidden_to_hidden,
            bias_input_to_hidden,
            bias_hidden_to_hidden,
        )
        np.testing.assert_array_equal(keras_lstm_weights[0], np.transpose(weight_input_to_hidden))
        np.testing.assert_array_equal(keras_lstm_weights[1], np.transpose(weight_hidden_to_hidden))
        np.testing.assert_array_equal(
            keras_lstm_weights[2], np.add(bias_input_to_hidden, bias_hidden_to_hidden)
        )
