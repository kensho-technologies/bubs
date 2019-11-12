# Copyright 2019 Kensho Technologies, LLC.
import unittest

import funcy
import numpy as np

from ..helpers import (
    InputEncoder, _align_sentence_spans_for_long_sentences, _check_token_spans, _pad_sentences,
    _reverse_inputs_and_indices, _shift_spans_to_start_at_zero, _split_long_sentences,
    create_document_indices_from_sentence_indices, get_space_joined_indices_from_token_lists,
    split_sentences_and_tokenize_raw_text
)


class TestHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.max_token_seq_len = 5
        cls.max_char_seq_len = 10
        cls.input_encoder = InputEncoder(cls.max_token_seq_len, cls.max_char_seq_len)

    def test_input_batches_from_raw_text(self):

        # Since this function mostly calls other functions, saving the detailed tests for the
        # other functions being called

        # Mind that max_tokens amd max_chars are set to small values for test
        batch_size = 128
        raw_text = (
            "S & P        Global Ratings and S & P Global\n Market Intelligence are owned by S & P "
            "Global Inc ."
        )
        expected_num_batches = 1
        # Important thing here is that whitespace between spans is correct
        expected_document_index_batches = [
            [
                [[0, 1], [2, 3], [4, 5]],  # S & P
                [[13, 19]],  # Global
                [[20, 27]],  # Ratings
                [[28, 31], [32, 33], [34, 35]],  # and S &
                [[36, 37], [38, 44]],  # P Global
                [[46, 52]],  # Market
                [[53, 61]],  # Intellig
                [[61, 65], [66, 69]],  # ence are
                [[70, 75], [76, 78]],  # owned by
                [[79, 80], [81, 82], [83, 84]],  # S & P
                [[85, 91]],  # Global
                [[92, 95], [96, 97]],  # Inc .
            ]
        ]
        expected_num_trimmed_sentences = len(expected_document_index_batches[0])
        gen, num_batches, index_batches = self.input_encoder.input_batches_from_raw_text(
            raw_text, batch_size=batch_size
        )
        self.assertEqual(expected_num_batches, num_batches)
        self.assertListEqual(expected_document_index_batches, index_batches)

        # Check that the shapes of all the inputs are correct
        for batch_index in range(num_batches):
            batch = next(gen)
            self.assertTupleEqual(
                batch["forward_input"].shape,
                (expected_num_trimmed_sentences, self.max_char_seq_len),
            )
            self.assertTupleEqual(
                batch["backward_input"].shape,
                (expected_num_trimmed_sentences, self.max_char_seq_len),
            )
            self.assertTupleEqual(
                batch["forward_index_input"].shape,
                (expected_num_trimmed_sentences, self.max_token_seq_len, 2),  # 2 for (row, col)
            )
            self.assertTupleEqual(
                batch["backward_index_input"].shape,
                (expected_num_trimmed_sentences, self.max_token_seq_len, 2),  # 2 for (row, col)
            )
            self.assertTupleEqual(
                batch["forward_mask_input"].shape,
                (expected_num_trimmed_sentences, self.max_token_seq_len),
            )
            self.assertTupleEqual(
                batch["backward_mask_input"].shape,
                (expected_num_trimmed_sentences, self.max_token_seq_len),
            )

        # Reduce batch size and check that there are more batches now
        batch_size = 5
        gen, num_batches, index_batches = self.input_encoder.input_batches_from_raw_text(
            raw_text, batch_size=batch_size
        )
        expected_num_batches = int(np.ceil(expected_num_trimmed_sentences / batch_size))
        self.assertEqual(expected_num_batches, num_batches)
        # Check that the shapes of all the inputs are correct

        # For all but the last batch, number of sentences should equal batch size.
        # Last batch should be shorter.
        for batch_index in range(num_batches - 1):
            batch = next(gen)
            self.assertEqual(batch["forward_input"].shape[0], batch_size)
        # last batch is going to be shorter
        last_batch = next(gen)
        self.assertEqual(
            last_batch["forward_input"].shape[0], expected_num_trimmed_sentences % batch_size
        )

    def test_input_batch_generator(self):
        # Throw in a few empty sentences
        tokenized_sentences = [["abab", "aaaa", "b", "ab", "."], [], []]
        batch_size = 2

        # We're expecting batches of size batch_size, in the order in which sentences are provided
        sentence_index_batches = funcy.chunks(batch_size, range(len(tokenized_sentences)))
        expected_generated_inputs = []

        # Compute what the generated input should be
        for sentence_index_batch in sentence_index_batches:
            tokenized_sentence_batch = [
                tokenized_sentences[index] for index in sentence_index_batch
            ]
            inputs_batch = self.input_encoder.prepare_inputs_from_pretokenized(
                tokenized_sentence_batch
            )
            expected_generated_inputs.append(inputs_batch)

        predict_generator = self.input_encoder.input_batch_generator(
            tokenized_sentences, batch_size
        )
        # verify the output of the generator
        for expected_inputs, inputs in zip(expected_generated_inputs, predict_generator):
            for key in expected_inputs:
                np.testing.assert_equal(expected_inputs[key], inputs[key])

        # now all empty sentences
        # test with emtpy sentences
        tokenized_sentences = [[]]
        batch_size = 4

        # We're expecting batches of size batch_size, in the order in which sentences are provided
        sentence_index_batches = funcy.chunks(batch_size, range(len(tokenized_sentences)))
        expected_generated_inputs = []

        # Compute what the generated input should be
        for sentence_index_batch in sentence_index_batches:
            tokenized_sentence_batch = [
                tokenized_sentences[index] for index in sentence_index_batch
            ]
            inputs_batch = self.input_encoder.prepare_inputs_from_pretokenized(
                tokenized_sentence_batch
            )
            expected_generated_inputs.append(inputs_batch)

        predict_generator = self.input_encoder.input_batch_generator(
            tokenized_sentences, batch_size
        )
        # verify the output of the generator
        for expected_inputs, inputs in zip(expected_generated_inputs, predict_generator):
            for key in expected_inputs:
                np.testing.assert_equal(expected_inputs[key], inputs[key])

    def test_prepare_inputs_from_pretokenized(self):
        tokenized_sentences = [["abb", "a", "."]]
        received_inputs = self.input_encoder.prepare_inputs_from_pretokenized(tokenized_sentences)
        expected_forward_input = np.array([[17, 6, 21, 21, 1, 6, 1, 18, 1]], dtype=np.int16)
        expected_backward_input = np.array([[17, 18, 1, 6, 1, 21, 21, 6, 1]], dtype=np.int16)
        expected_forward_index_input = np.array(
            [[[0, 0], [0, 0], [0, 4], [0, 6], [0, 8]]], dtype=np.int32
        )
        expected_backward_index_input = np.array(
            [[[0, 0], [0, 0], [0, 8], [0, 4], [0, 2]]], dtype=np.int32
        )
        expected_forward_mask_input = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        expected_backward_mask_input = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float64)

        np.testing.assert_equal(expected_forward_input, received_inputs["forward_input"])
        np.testing.assert_equal(expected_backward_input, received_inputs["backward_input"])
        np.testing.assert_equal(
            expected_forward_index_input, received_inputs["forward_index_input"]
        )
        np.testing.assert_equal(
            expected_backward_index_input, received_inputs["backward_index_input"]
        )
        np.testing.assert_equal(expected_forward_mask_input, received_inputs["forward_mask_input"])
        np.testing.assert_equal(
            expected_backward_mask_input, received_inputs["backward_mask_input"]
        )

        # test with an empty sentence
        tokenized_sentences = []
        received_inputs = self.input_encoder.prepare_inputs_from_pretokenized(tokenized_sentences)
        expected_forward_input = np.empty(shape=(0, 2), dtype=np.int16)
        expected_backward_input = np.empty(shape=(0, 2), dtype=np.int16)
        expected_forward_index_input = np.empty(shape=(0, 5, 2), dtype=np.int32)
        expected_backward_index_input = np.empty(shape=(0, 5, 2), dtype=np.int32)
        expected_forward_mask_input = np.empty(shape=(0, 5), dtype=np.float64)
        expected_backward_mask_input = np.empty(shape=(0, 5), dtype=np.float64)

        np.testing.assert_equal(expected_forward_input, received_inputs["forward_input"])
        np.testing.assert_equal(expected_backward_input, received_inputs["backward_input"])
        np.testing.assert_equal(
            expected_forward_index_input, received_inputs["forward_index_input"]
        )
        np.testing.assert_equal(
            expected_backward_index_input, received_inputs["backward_index_input"]
        )
        np.testing.assert_equal(expected_forward_mask_input, received_inputs["forward_mask_input"])
        np.testing.assert_equal(
            expected_backward_mask_input, received_inputs["backward_mask_input"]
        )

        # test with an empty token
        tokenized_sentences = [[""]]
        received_inputs = self.input_encoder.prepare_inputs_from_pretokenized(tokenized_sentences)
        expected_forward_input = np.array([[17, 1]], dtype=np.int16)
        expected_backward_input = np.array([[17, 1]], dtype=np.int16)
        expected_forward_index_input = np.array(
            [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]]], dtype=np.int32
        )
        expected_backward_index_input = np.array(
            [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]]], dtype=np.int32
        )
        expected_forward_mask_input = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        expected_backward_mask_input = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])

        np.testing.assert_equal(expected_forward_input, received_inputs["forward_input"])
        np.testing.assert_equal(expected_backward_input, received_inputs["backward_input"])
        np.testing.assert_equal(
            expected_forward_index_input, received_inputs["forward_index_input"]
        )
        np.testing.assert_equal(
            expected_backward_index_input, received_inputs["backward_index_input"]
        )
        np.testing.assert_equal(expected_forward_mask_input, received_inputs["forward_mask_input"])
        np.testing.assert_equal(
            expected_backward_mask_input, received_inputs["backward_mask_input"]
        )

    def test__encode_and_index(self):
        # test with one empty sentence
        tokenized_sentences = [[]]
        token_spans = [[]]
        (
            forward_inputs,
            backward_inputs,
            output_index_list_forward,
            output_index_list_backward,
        ) = self.input_encoder._encode_and_index(tokenized_sentences, token_spans)

        expected_forward_input = [[17, 1]]
        expected_backward_input = [[17, 1]]
        expected_forward_index_input = [[]]
        expected_backward_index_input = [[]]

        self.assertListEqual(expected_forward_input, forward_inputs)
        self.assertListEqual(expected_backward_input, backward_inputs)
        self.assertListEqual(expected_forward_index_input, expected_forward_index_input)
        self.assertListEqual(expected_backward_index_input, expected_backward_index_input)

        # test with completely empty inputs
        tokenized_sentences = []
        token_spans = []
        (
            forward_inputs,
            backward_inputs,
            output_index_list_forward,
            output_index_list_backward,
        ) = self.input_encoder._encode_and_index(tokenized_sentences, token_spans)
        expected_forward_input = []
        expected_backward_input = []
        expected_forward_index_input = []
        expected_backward_index_input = []

        self.assertListEqual(expected_forward_input, forward_inputs)
        self.assertListEqual(expected_backward_input, backward_inputs)
        self.assertListEqual(expected_forward_index_input, expected_forward_index_input)
        self.assertListEqual(expected_backward_index_input, expected_backward_index_input)

        # Test that an error is thrown if number of tokens is not equal to number of spans
        tokenized_sentences = [["a"] * 3, ["b"] * 2]
        token_spans = [[(0, 1)] * 3, [(0, 1)] * 3]

        with self.assertRaises(ValueError):
            self.input_encoder._encode_and_index(tokenized_sentences, token_spans)

        # Test that an error is thrown if len(tokenized_sentences) != len(token_spans)
        tokenized_sentences = [["a"] * 3, ["b"] * 2]
        token_spans = [[(0, 1)] * 3]

        with self.assertRaises(ValueError):
            self.input_encoder._encode_and_index(tokenized_sentences, token_spans)

        tokenized_sentences = [
            ["aa", "b", "a", "b", "."],
            ["aabb", "b", "."],
            ["", "", ""],
            ["abc", "."],
        ]

        # Checking that num_spans = num_tokens is enforced
        bad_token_spans = [
            [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 9)],  # here num of tokens != num of spans
            [(0, 5), (6, 7), (8, 9)],
            [(0, 0), (0, 0), (2, 2)],
            [(0, 3), (3, 4)],
        ]
        with self.assertRaises(ValueError):
            self.input_encoder._encode_and_index(tokenized_sentences, bad_token_spans)

        # Now check with correct inputs
        token_spans = [
            [(0, 2), (3, 4), (5, 6), (7, 8), (8, 9)],
            [(0, 4), (5, 6), (7, 8)],
            [(0, 0), (0, 0), (2, 2)],  # implying two spaces between '' and '' (must work anyway)
            [(0, 3), (3, 4)],
        ]
        expected_forw_encoded_sentences = [
            [17, 6, 6, 1, 21, 1, 6, 1, 21, 1],
            [17, 6, 6, 21, 21, 1, 21, 1, 18, 1],
            [17, 1, 1, 1],
            [17, 6, 21, 14, 18, 1],
        ]
        expected_back_encoded_sentences = [
            [17, 21, 1, 6, 1, 21, 1, 6, 6, 1],
            [17, 18, 1, 21, 1, 21, 21, 6, 6, 1],
            [17, 1, 1, 1],
            [17, 18, 14, 21, 6, 1],
        ]
        expected_output_index_list_forward = [[3, 5, 7, 9], [5, 7, 9], [1, 1, 3], [4, 5]]
        expected_output_index_list_backward = [[9, 6, 4, 2], [9, 4, 2], [3, 2, 2], [5, 1]]

        (
            forw_encoded_sentences,
            back_encoded_sentences,
            forw_index_list,
            back_index_list,
        ) = self.input_encoder._encode_and_index(tokenized_sentences, token_spans)

        self.assertListEqual(expected_forw_encoded_sentences, forw_encoded_sentences)
        self.assertListEqual(expected_back_encoded_sentences, back_encoded_sentences)
        self.assertListEqual(expected_output_index_list_forward, forw_index_list)
        self.assertListEqual(expected_output_index_list_backward, back_index_list)

    def test__check_labels(self):
        # make sure an error is thrown if labels are not one-hot-encoded
        bad_labels = np.array([[[0, 0, 0, 0], [0, 0, 0, 1]]])
        num_sentences = bad_labels.shape[0]
        with self.assertRaises(ValueError):
            self.input_encoder._check_labels(bad_labels, num_sentences)

        bad_labels = np.array([[[1, 1, 0, 0], [1, 0, 0, 0]]])
        num_sentences = bad_labels.shape[0]
        with self.assertRaises(ValueError):
            self.input_encoder._check_labels(bad_labels, num_sentences)

        # make sure an error is thrown if number of labels does not equal number of sentences
        bad_labels = np.array([[[1, 0, 0, 0], [1, 0, 0, 0]]])
        num_sentences = 1
        with self.assertRaises(ValueError):
            self.input_encoder._check_labels(bad_labels, num_sentences)

        # check completely empty labels - not properly one-hot encoded
        bad_labels = np.array([[[1, 0, 0, 0], [1, 0, 0, 0]]])
        num_sentences = 1
        with self.assertRaises(ValueError):
            self.input_encoder._check_labels(bad_labels, num_sentences)

        # Keep in mind: number of tokens is max_token_sequence_len
        # This should not raise any exceptions
        good_labels = np.array(
            [[[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]]
        )
        num_sentences = 1
        self.input_encoder._check_labels(good_labels, num_sentences)

    def test__pad_sentences(self):
        max_char_sequence_len = 10
        char_pad_value = 1
        encoded_sentence_list = [[4, 3, 2], [4, 1, 2, 3, 4], []]
        expected_padded_sentences = np.array(
            [
                [4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
                [4, 1, 2, 3, 4, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.int16,
        )
        padded_sentences = _pad_sentences(
            encoded_sentence_list, max_char_sequence_len, char_pad_value
        )
        np.testing.assert_array_equal(expected_padded_sentences, padded_sentences)

        encoded_sentence_list = [[]]
        expected_padded_sentences = np.array([[1] * max_char_sequence_len])
        padded_sentences = _pad_sentences(
            encoded_sentence_list, max_char_sequence_len, char_pad_value
        )
        np.testing.assert_array_equal(expected_padded_sentences, padded_sentences)

    def test__prepare_index_array(self):
        index_list_list = [[5, 3, 2], [0, 1, 2, 3, 4, 5], []]
        # (row, col) pairs, padded from the left
        expected_index_array = np.array(
            [
                [[0, 0], [0, 0], [0, 5], [0, 3], [0, 2]],
                [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
                [[2, 0], [2, 0], [2, 0], [2, 0], [2, 0]],
            ],
            dtype=np.int32,
        )
        index_array = self.input_encoder._prepare_index_array(index_list_list)
        np.testing.assert_array_equal(expected_index_array, index_array)

        index_list_list = [[]]
        expected_index_array = np.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]])
        index_array = self.input_encoder._prepare_index_array(index_list_list)
        np.testing.assert_array_equal(expected_index_array, index_array)

    def test__prepare_mask_array(self):
        index_list_list = [[5, 3, 2], [0, 1, 2, 3, 4], []]
        expected_mask_array = np.array(
            [[0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        mask_array = self.input_encoder._prepare_mask_array(index_list_list)
        np.testing.assert_array_equal(expected_mask_array, mask_array)

        index_list_list = [[]]
        expected_mask_array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        mask_array = self.input_encoder._prepare_mask_array(index_list_list)
        np.testing.assert_array_equal(expected_mask_array, mask_array)

    def test__encode_and_get_output_index_list(self):
        # 'a' is 6, 'b' is 21, '.' is 18
        token_list = ["aa", "b", "."]
        span_list = [(0, 2), (3, 4), (4, 5)]

        encoded_sentence, index_list = self.input_encoder._encode_and_get_output_index_list(
            token_list, span_list
        )

        expected_encoded_sentence = [17, 6, 6, 1, 21, 18, 1]
        expected_index_list = [3, 5, 6]

        self.assertListEqual(encoded_sentence, expected_encoded_sentence)
        self.assertListEqual(index_list, expected_index_list)

        # check that error is thrown if a span is shorter than the corresponding token
        token_list = ["aa", "b", "."]
        bad_span_list = [(0, 1), (1, 2), (2, 3)]
        with self.assertRaises(ValueError):
            self.input_encoder._encode_and_get_output_index_list(token_list, bad_span_list)

        # check that error is thrown if spans overlap
        token_list = ["aa", "b", "."]
        bad_span_list = [(0, 2), (1, 2), (2, 3)]
        with self.assertRaises(ValueError):
            self.input_encoder._encode_and_get_output_index_list(token_list, bad_span_list)

    def test__check_token_spans(self):
        # This should be fine
        word = "Bubbles"
        token_start = 0
        token_end = 7
        prev_index = 0
        _check_token_spans(word, token_start, token_end, prev_index)

        # Here token won't fit in its span
        word = "Bubbles"
        token_start = 0
        token_end = 6  # shorter than needed
        prev_index = 0
        with self.assertRaises(ValueError):
            _check_token_spans(word, token_start, token_end, prev_index)

        # Here token overlaps with previous token's end
        word = "Bubbles"
        token_start = 0
        token_end = 7  # shorter than needed
        prev_index = 1
        with self.assertRaises(ValueError):
            _check_token_spans(word, token_start, token_end, prev_index)

    def test__reverse_inputs_and_indices(self):
        # Boring sentence
        forward_sentence = [17, 6, 6, 1, 21, 1, 18, 1]
        forward_idx_list = [3, 5, 7]
        expected_backward_sentence = [17, 18, 1, 21, 1, 6, 6, 1]
        expected_backward_idx_list = [7, 4, 2]
        backward_sentence, backward_idx_list = _reverse_inputs_and_indices(
            forward_sentence, forward_idx_list
        )
        self.assertListEqual(expected_backward_sentence, backward_sentence)
        self.assertListEqual(expected_backward_idx_list, backward_idx_list)

        # Empty sentence
        forward_sentence = []
        forward_idx_list = []
        expected_backward_sentence = []
        expected_backward_idx_list = []
        backward_sentence, backward_idx_list = _reverse_inputs_and_indices(
            forward_sentence, forward_idx_list
        )
        self.assertListEqual(expected_backward_sentence, backward_sentence)
        self.assertListEqual(expected_backward_idx_list, backward_idx_list)

    def test_split_sentences_and_tokenize_raw_text(self):
        # A 'normal' case where no adjustments need to be made
        raw_text = "Yules went for a looooong run along the Charles in the morning. It was cold."
        max_tokens = 125
        max_chars = 2500
        processed_text = split_sentences_and_tokenize_raw_text(raw_text, max_tokens, max_chars)

        true_indices = [
            [
                (0, 5),
                (6, 10),
                (11, 14),
                (15, 16),
                (17, 25),
                (26, 29),
                (30, 35),
                (36, 39),
                (40, 47),
                (48, 50),
                (51, 54),
                (55, 62),
                (62, 63),
            ],
            [(0, 2), (3, 6), (7, 11), (11, 12)],
        ]
        true_tokens = [
            [
                "Yules",
                "went",
                "for",
                "a",
                "looooong",
                "run",
                "along",
                "the",
                "Charles",
                "in",
                "the",
                "morning",
                ".",
            ],
            ["It", "was", "cold", "."],
        ]
        self.assertListEqual(true_indices, processed_text["spans"])
        self.assertListEqual(true_tokens, processed_text["tokens"])

        # making max_tokens shorter than the sentences, but max_chars large enough
        raw_text = "Yules went for a looooong run along the Charles in the morning. It was cold."
        max_tokens = 5
        max_chars = 2500
        processed_text = split_sentences_and_tokenize_raw_text(raw_text, max_tokens, max_chars)

        true_indices = [
            [(0, 5), (6, 10), (11, 14), (15, 16), (17, 25)],
            [(0, 3), (4, 9), (10, 13), (14, 21), (22, 24)],
            [(0, 3), (4, 11), (11, 12)],
            [(0, 2), (3, 6), (7, 11), (11, 12)],
        ]
        true_tokens = [
            ["Yules", "went", "for", "a", "looooong"],
            ["run", "along", "the", "Charles", "in"],
            ["the", "morning", "."],
            ["It", "was", "cold", "."],
        ]
        self.assertListEqual(true_indices, processed_text["spans"])
        self.assertListEqual(true_tokens, processed_text["tokens"])

        # making max_chars comparably short to max_tokens
        raw_text = "Yules went for a looooong run along the Charles in the morning. It was cold."
        max_tokens = 5
        max_chars = 15
        processed_text = split_sentences_and_tokenize_raw_text(raw_text, max_tokens, max_chars)
        true_indices = [
            [(0, 5), (6, 10), (11, 14)],
            [(0, 1), (2, 10), (11, 14)],
            [(0, 5), (6, 9)],
            [(0, 7), (8, 10), (11, 14)],
            [(0, 7), (7, 8)],
            [(0, 2), (3, 6), (7, 11), (11, 12)],
        ]
        true_tokens = [
            ["Yules", "went", "for"],
            ["a", "looooong", "run"],
            ["along", "the"],
            ["Charles", "in", "the"],
            ["morning", "."],
            ["It", "was", "cold", "."],
        ]
        self.assertListEqual(true_indices, processed_text["spans"])
        self.assertListEqual(true_tokens, processed_text["tokens"])

        # check that a token longer than max_chars gets split up
        raw_text = "A looooooooooooooooong second token."
        max_tokens = 3
        max_chars = 10
        processed_text = split_sentences_and_tokenize_raw_text(raw_text, max_tokens, max_chars)
        true_indices = [[(0, 1)], [(0, 10)], [(0, 10)], [(0, 6)], [(0, 5), (5, 6)]]
        true_sentences = [["A"], ["looooooooo"], ["oooooooong"], ["second"], ["token", "."]]
        self.assertListEqual(true_indices, processed_text["spans"])
        self.assertListEqual(true_sentences, processed_text["tokens"])

        # Test extreme case where every token needs to be broken into individual characters
        raw_text = "A short sentence."
        max_tokens = 1
        max_chars = 2
        processed_text = split_sentences_and_tokenize_raw_text(raw_text, max_tokens, max_chars)
        true_indices = [
            [(0, 1)],
            [(0, 2)],
            [(0, 2)],
            [(0, 1)],
            [(0, 2)],
            [(0, 2)],
            [(0, 2)],
            [(0, 2)],
            [(0, 1)],
        ]
        true_sentences = [["A"], ["sh"], ["or"], ["t"], ["se"], ["nt"], ["en"], ["ce"], ["."]]
        self.assertListEqual(true_indices, processed_text["spans"])
        self.assertListEqual(true_sentences, processed_text["tokens"])

        # empty input should produce empty output
        raw_text = ""
        max_tokens = 3
        max_chars = 10
        processed_text = split_sentences_and_tokenize_raw_text(raw_text, max_tokens, max_chars)
        true_indices = [[]]
        true_sentences = [[]]
        self.assertListEqual(true_indices, processed_text["spans"])
        self.assertListEqual(true_sentences, processed_text["tokens"])

        # just a bunch of space characters
        raw_text = "  \n    \t"
        true_indices = [[]]
        true_sentences = [[]]
        processed_text = split_sentences_and_tokenize_raw_text(raw_text, max_tokens, max_chars)
        self.assertListEqual(true_indices, processed_text["spans"])
        self.assertListEqual(true_sentences, processed_text["tokens"])

    def test__align_sentence_spans_for_long_sentences(self):
        original_spans = [[(0, 1), (2, 9), (10, 15)], [(16, 23)]]
        # First sentence got split up here
        trimmed_spans = [[(0, 1), (2, 9)], [(0, 5)], [(0, 7)]]
        expected_aligned_original_spans = [[(0, 1), (2, 9)], [(10, 15)], [(16, 23)]]
        aligned_original_spans = _align_sentence_spans_for_long_sentences(
            original_spans, trimmed_spans
        )
        self.assertListEqual(expected_aligned_original_spans, aligned_original_spans)

        # If spans are identical to begin with, no changes need be made
        aligned_original_spans = _align_sentence_spans_for_long_sentences(
            original_spans, original_spans
        )
        self.assertListEqual(aligned_original_spans, original_spans)

        # Test empty sentence
        original_spans = []
        trimmed_spans = []
        expected_aligned_original_spans = [[]]
        aligned_original_spans = _align_sentence_spans_for_long_sentences(
            original_spans, trimmed_spans
        )
        self.assertListEqual(expected_aligned_original_spans, aligned_original_spans)

        # Test empty sentence
        original_spans = [[]]
        trimmed_spans = [[]]
        expected_aligned_original_spans = [[]]
        aligned_original_spans = _align_sentence_spans_for_long_sentences(
            original_spans, trimmed_spans
        )
        self.assertListEqual(expected_aligned_original_spans, aligned_original_spans)

    def test__shift_spans_to_start_at_zero(self):
        # not realistic because spans overlap, but it should handle this
        spans = [[(2, 3), (2, 3)]]
        expected_shifted_spans = [[(0, 1), (0, 1)]]
        shifted_spans = _shift_spans_to_start_at_zero(spans)
        self.assertListEqual(shifted_spans, expected_shifted_spans)

        spans = [[(-10, -20), (3, 5)]]
        expected_shifted_spans = [[(0, -10), (13, 15)]]
        shifted_spans = _shift_spans_to_start_at_zero(spans)
        self.assertListEqual(shifted_spans, expected_shifted_spans)

        spans = []
        expected_shifted_spans = []
        shifted_spans = _shift_spans_to_start_at_zero(spans)
        self.assertListEqual(shifted_spans, expected_shifted_spans)

        spans = [[]]
        expected_shifted_spans = [[]]
        shifted_spans = _shift_spans_to_start_at_zero(spans)
        self.assertListEqual(shifted_spans, expected_shifted_spans)

    def test__split_long_sentences(self):
        token_list_list = [
            ["Yules", "went", "for", "a", "looooong"],
            ["run", "along", "the", "Charles", "in"],
            ["the", "morning", "."],
            ["It", "was", "cold", "."],
        ]
        span_list_list = [
            [(0, 5), (6, 10), (11, 14), (15, 16), (17, 25)],
            [(0, 3), (4, 9), (10, 13), (14, 21), (22, 24)],
            [(0, 3), (4, 11), (11, 12)],
            [(0, 2), (3, 6), (7, 11), (11, 12)],
        ]

        # the case when sentences need to be cut shorter to fit in max_tokens
        max_tokens = 2
        max_chars = 20
        new_tokens, new_spans = _split_long_sentences(
            token_list_list, span_list_list, max_tokens, max_chars
        )

        expected_new_tokens = [
            ["Yules", "went"],
            ["for", "a"],
            ["looooong"],
            ["run", "along"],
            ["the", "Charles"],
            ["in"],
            ["the", "morning"],
            ["."],
            ["It", "was"],
            ["cold", "."],
        ]

        expected_new_spans = [
            [(0, 5), (6, 10)],
            [(0, 3), (4, 5)],
            [(0, 8)],
            [(0, 3), (4, 9)],
            [(0, 3), (4, 11)],
            [(0, 2)],
            [(0, 3), (4, 11)],
            [(0, 1)],
            [(0, 2), (3, 6)],
            [(0, 4), (4, 5)],
        ]
        self.assertListEqual(expected_new_tokens, new_tokens)
        self.assertListEqual(expected_new_spans, new_spans)

        # when max_chars is more restrictive than max_tokens
        max_tokens = 10
        max_chars = 10

        new_tokens, new_spans = _split_long_sentences(
            token_list_list, span_list_list, max_tokens, max_chars
        )

        expected_new_tokens = [
            ["Yules", "went"],
            ["for", "a"],
            ["looooong"],
            ["run", "along"],
            ["the"],
            ["Charles", "in"],
            ["the"],
            ["morning", "."],
            ["It", "was"],
            ["cold", "."],
        ]
        expected_new_spans = [
            [(0, 5), (6, 10)],
            [(0, 3), (4, 5)],
            [(0, 8)],
            [(0, 3), (4, 9)],
            [(0, 3)],
            [(0, 7), (8, 10)],
            [(0, 3)],
            [(0, 7), (7, 8)],
            [(0, 2), (3, 6)],
            [(0, 4), (4, 5)],
        ]
        self.assertListEqual(expected_new_tokens, new_tokens)
        self.assertListEqual(expected_new_spans, new_spans)

        # when no changes need to be made
        max_tokens = 50
        max_chars = 200
        new_tokens, new_spans = _split_long_sentences(
            token_list_list, span_list_list, max_tokens, max_chars
        )
        expected_new_tokens = [
            ["Yules", "went", "for", "a", "looooong"],
            ["run", "along", "the", "Charles", "in"],
            ["the", "morning", "."],
            ["It", "was", "cold", "."],
        ]
        expected_new_spans = [
            [(0, 5), (6, 10), (11, 14), (15, 16), (17, 25)],
            [(0, 3), (4, 9), (10, 13), (14, 21), (22, 24)],
            [(0, 3), (4, 11), (11, 12)],
            [(0, 2), (3, 6), (7, 11), (11, 12)],
        ]
        self.assertListEqual(expected_new_tokens, new_tokens)
        self.assertListEqual(expected_new_spans, new_spans)

        # empty input
        max_tokens = 5
        max_chars = 5
        token_list_list = []
        span_list_list = []
        new_tokens, new_spans = _split_long_sentences(
            token_list_list, span_list_list, max_tokens, max_chars
        )
        expected_new_tokens = [[]]
        expected_new_spans = [[]]
        self.assertListEqual(expected_new_tokens, new_tokens)
        self.assertListEqual(expected_new_spans, new_spans)

        # test that a ValueError is raised when max_chars or max_tokens is less than 1
        max_tokens = 0
        max_chars = 20
        with self.assertRaises(ValueError):
            _split_long_sentences(token_list_list, span_list_list, max_tokens, max_chars)

        max_tokens = 10
        max_chars = -2
        with self.assertRaises(ValueError):
            _split_long_sentences(token_list_list, span_list_list, max_tokens, max_chars)

    def test_get_space_joined_indices_from_token_list(self):
        token_lists = [["Ray", "walked", "his", "dog", "."]]
        spans = get_space_joined_indices_from_token_lists(token_lists)
        expected_spans = [[(0, 3), (4, 10), (11, 14), (15, 18), (19, 20)]]
        self.assertEqual(spans, expected_spans)

        token_lists = [
            ["Ray", "walked", "his", "dog", "."],
            ["Bubbles", "is", "a", "good", "doggo", "."],
        ]
        spans = get_space_joined_indices_from_token_lists(token_lists)
        expected_spans = [
            [(0, 3), (4, 10), (11, 14), (15, 18), (19, 20)],
            [(0, 7), (8, 10), (11, 12), (13, 17), (18, 23), (24, 25)],
        ]
        self.assertEqual(spans, expected_spans)

        token_lists = [[]]
        spans = get_space_joined_indices_from_token_lists(token_lists)
        expected_spans = [[]]
        self.assertEqual(spans, expected_spans)

        token_lists = []
        spans = get_space_joined_indices_from_token_lists(token_lists)
        expected_spans = []
        self.assertEqual(spans, expected_spans)

    def test_create_document_indices_from_sentence_indices(self):
        # Take a few sentences, tokenize. Then align back to the raw text
        raw_text = (
            "S & P        Global Ratings and S & P Global Market Intelligence"
            ' are owned by S & P Global Inc.\n  The rating was assessed to be " improving . "'
        )
        max_tokens = 3
        max_chars = 10
        processed_text = split_sentences_and_tokenize_raw_text(raw_text, max_tokens, max_chars)
        tokens = processed_text["tokens"]
        spans = processed_text["spans"]

        document_indices = create_document_indices_from_sentence_indices(spans, tokens, raw_text)
        # check that tokens are actually in the raw_text at the correct locations
        for idx_list, token_list in zip(document_indices, tokens):
            for span, token in zip(idx_list, token_list):
                self.assertEqual(token, raw_text[span[0]: span[1]])

        # Test for document that starts with white space - make sure document spans are
        # appropriately shifted
        raw_text = "\n\r The United States is a place."
        sentence_list = [["The", "United", "States", "is", "a", "place", "."]]
        span_list = [[(0, 3), (4, 10), (11, 17), (18, 20), (21, 22), (23, 28), (28, 29)]]
        expected_document_spans = [
            [[3, 6], [7, 13], [14, 20], [21, 23], [24, 25], [26, 31], [31, 32]]
        ]
        received_spans = create_document_indices_from_sentence_indices(
            span_list, sentence_list, raw_text
        )
        self.assertEqual(expected_document_spans, received_spans)
