# Copyright 2019 Kensho Technologies, LLC.
from funcy import chunks, first, last
import numpy as np

from .char_to_int import CHAR_TO_INT
from .tokenizer import RegexTokenizer


class InputEncoder:
    """Object that prepares inputs to the ContextualizedEmbedding layer from text."""

    def __init__(self, max_token_sequence_len, max_char_sequence_len):
        """Initialize and look up a few special character codes from CHAR_TO_INT."""
        self.max_token_sequence_len = max_token_sequence_len
        self.max_char_sequence_len = max_char_sequence_len

        # Parameters related to character encoding, all from char_to_int
        self.char_to_int = CHAR_TO_INT

    @property
    def start_sentence_value(self,):
        """Each encoded sentence will start with this value."""
        return self.char_to_int["\n"]

    @property
    def space_char_code(self,):
        """In the model input, all tokens will be separated by exactly one space character code."""
        return self.char_to_int[" "]

    @property
    def unk_char_code(self,):
        """Every character absent from our char_to_int dict will be mapped onto this code."""
        return self.char_to_int["<unk>"]

    @property
    def end_sentence_value(self,):
        """We will append this value to the end of each sentence."""
        return self.char_to_int[" "]

    @property
    def char_pad_value(self,):
        """Short sentences will be padded with this value."""
        return self.char_to_int[" "]

    def input_batches_from_raw_text(self, raw_text, batch_size=32):
        """Construct ContextualizedEmbedding inputs from a raw text string.

        Args:
            raw_text: str, the entire raw document
            batch_size: number of sentences per batch; limited by GPU memory

        Returns:
            generator: yields dicts of inputs to the ContextualizedEmbedding layer
                'forward_input': padded array of character codes corresponding to each sentence
                'backward_input': padded array of character codes in reverse order
                'forward_index_input': padded array of locations of token outputs in forward_input
                'backward_index_input': padded array of locations of token outputs in backward_input
                'forward_mask_input': mask of same shape as forward_index_input, with 0's where
                    padded and 1's where real tokens
                'backward_mask_input':mask of same shape as back_index_input, with 0's where
                    padded and 1's where real tokens
            num_batches: int, necessary because after enforcing sentence limits we may end up with
                more sentences than we expect
            document_index_batches: list of lists of (start, end) tuples indicating spans where each
                token came from in raw_text. Outer list over batches, inner list over sentences
        """
        parsed_text = split_sentences_and_tokenize_raw_text(
            raw_text, self.max_token_sequence_len, self.max_char_sequence_len - 2
        )  # -2 for special start and end characters

        generator = self.input_batch_generator(parsed_text["tokens"], batch_size)
        num_batches = int(np.ceil(len(parsed_text["tokens"]) / batch_size))
        document_indices = create_document_indices_from_sentence_indices(
            parsed_text["spans"], parsed_text["tokens"], raw_text
        )
        # now group document_indices into batches
        document_index_batches = list(chunks(batch_size, document_indices))
        return generator, num_batches, document_index_batches

    def input_batch_generator(self, tokenized_sentences, batch_size):
        """Yield inputs to ContextualizedEmbedding in batches with minimal padding for prediction.

        Group sentences into batches in the order they're provided. Character-level padding is
        determined by longest sentence in the batch. Yield one batch at a time.

        Args:
            tokenized_sentences: list of lists of str, each str a token
            batch_size: int, number of sentences per batch generated

        Returns:
            Yields inputs to ContextualizedEmbedding one sentence batch at a time
        """
        while True:
            for chunk in chunks(batch_size, range(len(tokenized_sentences))):
                selected_sentences = [tokenized_sentences[index] for index in chunk]
                model_inputs = self.prepare_inputs_from_pretokenized(selected_sentences)
                yield model_inputs

    def prepare_inputs_from_pretokenized(self, tokenized_sentences):
        """Construct inputs to ContextualizedEmbedding from tokenized sentences and optional labels.

        Character-level padding length is determined by the longest sentence in the batch.
        Use at your own risk. This assumes the sentences already fit into character and token
        limits.

        Args:
            tokenized_sentences: List of lists of str, one str for each token
        Returns:
            a dict of inputs to the ContextualizedEmbedding layer
                'forward_input': padded array of character codes corresponding to each sentence
                'backward_input': padded array of character codes in reverse order
                'forward_index_input': padded array of locations of token outputs in forward_input
                'backward_index_input': padded array of locations of token outputs in backward_input
                'forward_mask_input': mask of same shape as forward_index_input, with 0's where
                    padded and 1's where real tokens
                'backward_mask_input':mask of same shape as back_index_input, with 0's where
                    padded and 1's where real tokens
        """
        token_spans = get_space_joined_indices_from_token_lists(tokenized_sentences)

        # Pad everything to longest sentence length
        sentence_lengths = [last(spans)[1] for spans in token_spans if len(spans) > 0]
        longest_sentence_len = max(sentence_lengths + [0])
        pad_len = longest_sentence_len + 2  # 2 for extra start character and end character

        # encode sentences and get output indices (located at token edges)
        (
            forward_inputs,
            backward_inputs,
            output_index_list_forward,
            output_index_list_backward,
        ) = self._encode_and_index(tokenized_sentences, token_spans)

        # pad sentences
        forward_inputs = _pad_sentences(forward_inputs, pad_len, self.char_pad_value)
        backward_inputs = _pad_sentences(backward_inputs, pad_len, self.char_pad_value)

        # Make inputs used in indexing and masking
        forward_index_array = self._prepare_index_array(output_index_list_forward)
        backward_index_array = self._prepare_index_array(output_index_list_backward)
        forward_mask = self._prepare_mask_array(output_index_list_forward)
        backward_mask = self._prepare_mask_array(output_index_list_backward)

        model_inputs = {
            "forward_input": forward_inputs,
            "backward_input": backward_inputs,
            "forward_index_input": forward_index_array,
            "backward_index_input": backward_index_array,
            "forward_mask_input": forward_mask,
            "backward_mask_input": backward_mask,
        }

        return model_inputs

    def _encode_and_index(self, tokenized_sentences, token_spans):
        """Encode a list of tokenized sentences and record appropriate output indices.

        Translate sentences into forward and backward lists of character codes with appropriate
        start and end values; return also the locations where token-level embeddings will be
        outputted (token boundaries) relative to forward and backward encoded sentences
        """
        forward_inputs = []
        backward_inputs = []
        output_index_list_forward = []
        output_index_list_backward = []

        if len(tokenized_sentences) != len(token_spans):
            raise ValueError(
                "Number of tokenized sentences does not match number of token span "
                "lists.\nNumber of sentences: {}\n"
                "Number of token span lists: {}".format(len(tokenized_sentences), len(token_spans))
            )

        # Translate characters in each sentence to numbers using character_code_dict
        for sentence, token_span_list in zip(tokenized_sentences, token_spans):
            forw_sentence, forw_index_list = self._encode_and_get_output_index_list(
                sentence, token_span_list
            )
            back_sentence, back_index_list = _reverse_inputs_and_indices(
                forw_sentence, forw_index_list
            )
            forward_inputs.append(forw_sentence)
            backward_inputs.append(back_sentence)
            output_index_list_forward.append(forw_index_list)
            output_index_list_backward.append(back_index_list)
        return (
            forward_inputs,
            backward_inputs,
            output_index_list_forward,
            output_index_list_backward,
        )

    def _check_labels(self, labels, num_sentences):
        """Check that labels are one-hot encoded and have correct shape."""
        max_token_sequence_len = self.max_token_sequence_len
        labels_shape = labels.shape
        if not (labels.sum(axis=2) == 1).all():
            raise ValueError("Labels are not provided in the correct one-hot-encoded format")

        if labels_shape[0] != num_sentences:
            raise ValueError(
                "Shape of label array does not agree with number of tokenized sentences"
            )

        if labels_shape[1] != self.max_token_sequence_len:
            raise ValueError(
                "Label array not padded to correct token sequence length.\n"
                "Expected token sequence length: {}\n"
                "Discovered label array shape: {}".format(max_token_sequence_len, labels_shape)
            )

    def _prepare_index_array(self, index_list):
        """Make a 2D array where each row is a padded array of character-level token-end indices."""
        pad_len = self.max_token_sequence_len
        batch_size = len(index_list)
        padding_index = 0
        padded_sentences = np.full((batch_size, pad_len, 2), padding_index, dtype=np.int32)
        for i in range(batch_size):
            clipped_len = min(len(index_list[i]), pad_len)
            padded_sentences[i, :, 0] = i
            padded_sentences[i, pad_len - clipped_len:, 1] = index_list[i][:clipped_len]
        return padded_sentences

    def _prepare_mask_array(self, index_list):
        """Make 2D array where each row contains 1's where real tokens were and 0's where padded."""
        pad_len = self.max_token_sequence_len
        batch_size = len(index_list)
        mask = np.zeros((batch_size, pad_len))
        for i, inds in enumerate(index_list):
            mask[i, pad_len - len(inds):] = 1
        return mask

    def _encode_and_get_output_index_list(self, token_list, span_list):
        """Transform a tokenized sentence into lists of character codes and lists of indices.

        Each encoded sentence starts with a special start symbol. Forward and backward sentences
        are reverses of each other (except for the start symbol). If tokenized sentence exceeds
        max number of tokens of characters, it is trimmed.

        Args:
            token_list: list of str, one for each token
            span_list: list of tuples (start, end) of token locations in the sentence. Anything
                outside of the (start, end) spans is assumed to be a space

        Returns:
            encoded_sentence_forward: array of character codes starting with a special start symbol
            output_index_list_forw: list of indices of token ends (accounting for spaces and special
                start symbol) as they appear in encoded_sentence_forward
        """
        #  Make sure that number of tokens matches number of spans
        if len(token_list) != len(span_list):
            raise ValueError(
                "Number of tokens does not match number of token spans.\n"
                "Tokens provided: {},\n"
                "Token spans provided: {}".format(token_list, span_list)
            )

        encoded_sentence_forward = [self.start_sentence_value]
        output_index_list_forw = []
        prev_index = 0  # index where previous token ended, for keeping track of whitespace

        for token, (token_start, token_end) in zip(
            token_list[: self.max_token_sequence_len], span_list[: self.max_token_sequence_len]
        ):

            _check_token_spans(token, token_start, token_end, prev_index)
            # proceed only if we fit into the character limit
            if token_end >= self.max_char_sequence_len - 1:
                break
            num_spaces = token_start - prev_index
            encoded_sentence_forward.extend([self.space_char_code] * num_spaces)
            output_index_list_forw.append(token_end + 1)  # +1 for start_sentence_value
            for char in token:
                encoded_sentence_forward.append(self.char_to_int.get(char, self.unk_char_code))
            prev_index = token_end

        encoded_sentence_forward.append(self.end_sentence_value)
        return encoded_sentence_forward, output_index_list_forw


def _pad_sentences(encoded_sentence_list, pad_len, char_pad_value):
    """Take in a list of lists of character codes, return padded 2D array."""
    padded_sentences = np.full(
        (len(encoded_sentence_list), pad_len), char_pad_value, dtype=np.int16
    )
    for sentence_ind, encoded_sentence in enumerate(encoded_sentence_list):
        padded_sentences[sentence_ind, : len(encoded_sentence)] = encoded_sentence
    return padded_sentences


def _check_token_spans(token, token_start, token_end, prev_index):
    """Check that token fits into its span; output a helpful message is it doesn't."""
    if len(token) > (token_end - token_start):
        raise ValueError(
            "Token is longer than its allocated span.\n"
            "Token: {}\n"
            "Span: ({}, {})".format(token, token_start, token_end)
        )

    # check that spans don't overlap
    if token_start < prev_index:
        raise ValueError(
            "Overlapping token spans.\n"
            "Token: {}\n"
            "Span: ({}, {})".format(token, token_start, token_end)
        )


def _reverse_inputs_and_indices(encoded_sentence_forward, output_index_list_forward):
    """Reverse sequence of character codes and list of output indices."""
    if len(encoded_sentence_forward) >= 2:  # sentence should at least have start, end characters
        start_sentence_value = first(encoded_sentence_forward)
        end_sentence_value = last(encoded_sentence_forward)
        encoded_sentence_length = len(encoded_sentence_forward)

        # Reverse all character codes in the sentence without affecting the first and last elements
        # (those are special start_sentence_value and end_sentence_value)
        encoded_sentence_back = [start_sentence_value]
        encoded_sentence_back.extend(encoded_sentence_forward[-2:0:-1])  # skip start and end
        encoded_sentence_back.append(end_sentence_value)
    else:
        encoded_sentence_back = []

    # compute backward output indices
    if len(output_index_list_forward) == 0:
        locations_before_tokens = []
    else:
        locations_before_tokens = [0] + output_index_list_forward[:-1]
    output_indices_back = [encoded_sentence_length - x - 1 for x in locations_before_tokens]
    return encoded_sentence_back, output_indices_back


def split_sentences_and_tokenize_raw_text(raw_text, max_token_sequence_len, max_char_sequence_len):
    """Tokenize raw text into lists of tokens.

    Further ensure that no resulting sentence will exceed the specified token limit or character
    limit. If any single token by itself exceeds the sentence character limit, it will be split into
    tokens of max_char_sequence_len characters.

    Args:
        raw_text: str, the entire text document to be turned into batches of sentences
        max_token_sequence_len: maximum number of tokens per sentence
        max_char_sequence_len: maximum number of characters per sentence

    Returns:
        a dict containing:
            tokens: list of lists of str, one str per token
            spans: list of lists of tuples (start, end) of character-level token locations.
                Note that these are adjusted so each sentence starts at zero, and all tokens are
                separated by one space.
    """
    tokenizer = RegexTokenizer(max_characters_per_token=max_char_sequence_len)
    token_lists, spans_in_original_text = tokenizer.tokenize(raw_text)

    # Compute (start, end) character-level spans corresponding to each token
    space_added_spans = get_space_joined_indices_from_token_lists(token_lists)

    # Enforce character-level and token-level sentence length limits
    adjusted_token_lists, adjusted_span_lists = _split_long_sentences(
        token_lists, space_added_spans, max_token_sequence_len, max_char_sequence_len
    )

    # Now adjust spans_in_original_text to break up longer sentences
    trimmed_original_spans = _align_sentence_spans_for_long_sentences(
        spans_in_original_text, adjusted_span_lists
    )

    # shift all spans to 0 for consistency
    shifted_original_spans = _shift_spans_to_start_at_zero(trimmed_original_spans)

    return {"tokens": adjusted_token_lists, "spans": shifted_original_spans}


def _align_sentence_spans_for_long_sentences(original_sentence_spans, trimmed_sentence_spans):
    """Align new token spans after enforcing limits to the original locations in raw text.

    This is needed to keep track of each token's location in original document. After enforcing
    sentence limits, some sentences get split into multiple parts. The locations in original
    document don't change, but we need to maintain one list per sentence. So we regroup the spans
    into lists corresponding to the trimmed sentences.

    Args:
        original_sentence_spans: list of lists of (start, end) int tuples that came out of the
            tokenizer. All locations with respect to the beginning of the document.
        trimmed_sentence_spans: list of lists of (start, end) int tuples, each sentence starting at
            zero, after splitting any long sentences into multiple chunks.

    Returns:
        adjusted spans: spans pointing to the original document, but regrouped into new lists
            anytime a sentence was split.
    """
    if len(original_sentence_spans) == 0:
        return [[]]
    original_sentence_index = 0
    sentence_break = 0
    adjusted_spans = []
    for trimmed_sentence in trimmed_sentence_spans:
        original_sentence = original_sentence_spans[original_sentence_index]
        if len(trimmed_sentence) < len(original_sentence):
            new_sentence_break = sentence_break + len(trimmed_sentence)
            adjusted_spans.append(original_sentence[sentence_break:new_sentence_break])
            if new_sentence_break == len(original_sentence):
                sentence_break = 0
                original_sentence_index += 1
            else:
                sentence_break = new_sentence_break
        else:
            adjusted_spans.append(original_sentence)
            original_sentence_index += 1
    return adjusted_spans


def _shift_spans_to_start_at_zero(spans):
    """Shift all spans in the sentence by the same amount so the first token starts at zero.

    Args:
        spans: list of lists of character-level spans, one span per token, one list per sentence

    Returns:
        list of list of spans shifted so that first token in each sentence starts at zero
    """
    adjusted_spans = []
    for span_list in spans:
        if len(span_list) > 0:
            offset = first(span_list)[0]
            adjusted_spans.append([(span[0] - offset, span[1] - offset) for span in span_list])
        else:
            adjusted_spans.append([])
    return adjusted_spans


def _split_long_sentences(token_lists, span_lists, max_tokens, max_chars):
    """Split tokenized sentences to enforce max_tokens and max_chars. Adjust character spans.

    This function does not enforce correct token length: it assumes that any individual token is
    shorter than max_chars characters.

    Args:
        token_lists: List of lists of str (one str per token). Each token shorter than max_chars
        span_lists: List of lists of tuples of (start, end) token locations in sentence
        max_tokens: maximum number of tokens per sentence
        max_chars: maximum number of characters per sentence

    Returns:
        new_tokens: new list of lists of str, where each list conforms to max_chars and max_tokens
        new_spans: new list of lists of (start, end) tuples. Spans start at 0 for each sentence
    """
    if max_tokens < 1 or max_chars < 1:
        raise ValueError(
            "Expected max_tokens and max_chars to be at least 1\n"
            "Found max_tokens = {}, max_chars = {}".format(max_tokens, max_chars)
        )

    new_tokens = []
    new_spans = []
    for token_list, span_list in zip(token_lists, span_lists):
        # check for bad input
        if len(token_list) != len(span_list):
            raise ValueError(
                "Bad tokenized sentence: number of "
                "tokens does not equal number of spans\n"
                "tokens: {}\ntoken spans: {}".format(token_list, span_list)
            )
        # check if empty
        if len(token_list) == 0:
            continue

        # check if we already fit into the limits to avoid unnecessary loops
        if len(token_list) <= max_tokens and last(span_list)[1] <= max_chars:
            new_tokens.append(token_list)
            new_spans.append(span_list)
            continue

        # loop over tokens until fill up a sentence chunk
        token_index = 0
        token_shift = 0
        span_shift = 0
        sentence_token_chunk = []
        sentence_span_chunk = []

        while token_index < len(token_list):
            token = token_list[token_index]
            span = span_list[token_index]

            # if token fits in
            if span[1] - span_shift <= max_chars and token_index - token_shift < max_tokens:
                sentence_token_chunk.append(token)
                adjusted_span = (span[0] - span_shift, span[1] - span_shift)
                sentence_span_chunk.append(adjusted_span)

            # if token doesn't fit, need to start a new sentence chunk
            else:
                if len(sentence_token_chunk) > 0:
                    new_tokens.append(sentence_token_chunk)
                    new_spans.append(sentence_span_chunk)
                span_shift = span[0]
                token_shift = token_index
                sentence_token_chunk = [token]
                sentence_span_chunk = [(0, span[1] - span[0])]
            token_index += 1

        new_tokens.append(sentence_token_chunk)
        new_spans.append(sentence_span_chunk)
    if len(new_tokens) == 0:
        # preserve the dimensionality of the output
        new_tokens = [[]]
        new_spans = [[]]
        return new_tokens, new_spans
    return new_tokens, new_spans


def get_space_joined_indices_from_token_lists(token_lists):
    """Compute character-level locations of a set of tokens assuming that they are separated by ' '.

    Args:
        token_lists: list of list of str

    Returns:
        span_lists: list of list of integer tuples (start, end)
    """
    token_lengths = [[len(token) for token in sentence] for sentence in token_lists]

    token_ends = []
    for sentence in token_lengths:
        sentence_token_ends = list(np.array(sentence).cumsum() + range(len(sentence)))
        token_ends.append(list(map(int, sentence_token_ends)))

    token_starts = []
    for sentence1, sentence2 in zip(token_ends, token_lengths):
        token_starts.append(
            [token_end - token_length for token_end, token_length in zip(sentence1, sentence2)]
        )
    span_lists = [
        list(zip(sentence1, sentence2)) for sentence1, sentence2 in zip(token_starts, token_ends)
    ]
    return span_lists


def create_document_indices_from_sentence_indices(span_lists, token_lists, document):
    """Convert sentence spans (each sentence starting at 0) to document spans.

    Args:
        span_lists: list of lists of tuples of int (start, end) representing token locs
        token_lists: list of list of str. sentences and tokens in document.
        document: str. raw text of predicted document

    Returns:
        document_span_lists: A list of lists of tuples of int (start, end)
    """
    sentence_lengths = [last(span_list)[-1] for span_list in span_lists]
    sentence_starts = []
    offset = 0

    # We have to base our location off of the original document to deal with weird sentences
    # For example: "Yuliya loves cats.    Ray loves dogs." or the case where as sentence is split
    # Mid-word due to exceeding the max sentence list
    # We select the first length, and the second sentence and so on to get the offsets
    for length, token_list in zip(sentence_lengths, token_lists):
        next_start = document[offset:].find(first(token_list))
        offset = offset + next_start
        sentence_starts.append(offset)
        offset = offset + length

    # Modify our sentence indices so that the sentences line up with the original text
    document_span_lists = []
    for start, span_list in zip(sentence_starts, span_lists):
        document_span_lists.append(
            [[span_start + start, span_end + start] for (span_start, span_end) in span_list]
        )

    return document_span_lists
