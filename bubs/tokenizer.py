# Copyright 2019 Kensho Technologies, LLC.
import funcy
from segtok.segmenter import split_single, to_unix_linebreaks
from segtok.tokenizer import (
    split_contractions, split_possessive_markers, web_tokenizer, word_tokenizer
)


def _html_tokenize(sentence):
    """Tokenize string into words, not splitting URIS or emails, wrapping segtok:word_tokenizer.

    It does not split URIs or e-mail addresses. It does not treat html escapes
    as single characters outside of these instances.(eg. &amp; -> '&', 'amp', ';')

    Args:
        sentence: input string for tokenization

    Returns:
        tokens: list of str
    """
    tokens = []
    for i, span in enumerate(web_tokenizer.split(sentence)):
        if i % 2:
            tokens.append(span)
        else:
            tokens.extend(word_tokenizer(span))
    return tokens


class RegexTokenizer:
    """Fast regex-based tokenizer, wrapping around https://github.com/fnl/segtok ."""

    def __init__(self, max_characters_per_token=None):
        """Set up tokenizer specific flags.

        Args:
            max_characters_per_token: maximum allowed token length, token will be split if too long
        """
        if max_characters_per_token is not None and max_characters_per_token < 1:
            raise ValueError(
                "Maximum number of characters per token cannot be less than 1.\n"
                "Max characters requested:{}".format(max_characters_per_token)
            )
        self._max_characters_per_token = max_characters_per_token

    def word_tokenize(self, text):
        """Get list of string tokens from input string.

        Args:
            text: input string for tokenization
        Yields:
            token: str, non-whitespace tokens
        """
        for token in split_possessive_markers(split_contractions(_html_tokenize(text))):
            if self._max_characters_per_token is not None:
                for token_chunk in funcy.chunks(self._max_characters_per_token, token):
                    yield token_chunk
            else:
                yield token

    def sentence_tokenize(self, text):
        """Get list of string sentences from input string.

        Args:
            text: raw input string
        Yields:
            str: non-whitespace, non-empty sentence strings
        """
        for sentence in split_single(to_unix_linebreaks(text)):
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 0:
                yield clean_sentence

    def tokenize(self, text):
        """Tokenize string into sentences and words and return list of lists of str.

        Args:
            text: raw input string
        Returns:
            tokenized_text: list of str tokens
            token_spans: list of (start, end) tuples specifying token locations in text
        """
        tokenized_text = []
        token_spans = []
        for sentence in self.sentence_tokenize(text):
            tokenized_sentence = list(self.word_tokenize(sentence))
            # further split if we have a max sentence length
            tokenized_text.append(tokenized_sentence)
            token_spans.append(_align_multi_token_lists(sentence, [tokenized_sentence]))
        return tokenized_text, token_spans


def _align_multi_token_lists(text, tokenized_document):
    """Align lists of tokenized sentences to text and return lists of token spans."""
    char_offset = 0
    token_spans = []
    first_sentence = True
    for tokenized_sentence in tokenized_document:
        # use first sentence indicator for adding sentence boundary tokens
        if first_sentence:
            first_sentence = False
        char_offset, parsed_sentence_tokens = _align_token_list(
            text, tokenized_sentence, char_offset
        )
        token_spans.extend(parsed_sentence_tokens)
    # add trailing whitespace token if necessary
    if char_offset < len(text):
        token_spans.append((char_offset, char_offset + len(text)))
    return token_spans


def _align_token_list(text, token_list, char_offset=0):
    """Align list of string tokens to text and return list of Token objects."""
    token_spans = []
    for text_token in token_list:
        start = text.index(text_token, char_offset)
        token_spans.append((start, start + len(text_token)))
        char_offset = start + len(text_token)
    return char_offset, token_spans
