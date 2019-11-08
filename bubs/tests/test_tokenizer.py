# Copyright 2019 Kensho Technologies, LLC.
import unittest

from ..tokenizer import RegexTokenizer


class TestRegexTokenizer(unittest.TestCase):
    def test_regex_tokenizer(self):
        tokenizer = RegexTokenizer()
        # Simple tokenization
        text = "Muffins are delicious."
        expected_tokens = [["Muffins", "are", "delicious", "."]]
        expected_spans = [[(0, 7), (8, 11), (12, 21), (21, 22)]]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)

        # test long sentence
        text = "catch my tail " * 100
        expected_tokens = [["catch", "my", "tail"] * 100]
        tokens, _ = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)

        # multiple sentences
        text = "Muffins are delicious. I lost my hat."
        expected_tokens = [["Muffins", "are", "delicious", "."], ["I", "lost", "my", "hat", "."]]
        expected_spans = [
            [(0, 7), (8, 11), (12, 21), (21, 22)],
            [(0, 1), (2, 6), (7, 9), (10, 13), (13, 14)],
        ]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # linebreaks also split
        text = "Muffins are\ndelicious."
        expected_tokens = [["Muffins", "are"], ["delicious", "."]]
        expected_spans = [[(0, 7), (8, 11)], [(0, 9), (9, 10)]]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # unicode linebreaks as well
        text = "Muffins are\x0adelicious."
        expected_tokens = [["Muffins", "are"], ["delicious", "."]]
        expected_spans = [[(0, 7), (8, 11)], [(0, 9), (9, 10)]]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # edge cases don't raise
        text = ""
        expected_tokens = []
        expected_spans = []
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # proper whitespace stripping including unicode
        text = "\n\n\t\x0a\t \u3000\t"
        expected_tokens = []
        expected_spans = []
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # combination of many things
        text = " \n ''Sentence\" \tone. Another \u201cWEIRD sentence  !\t\t"
        expected_tokens = [
            ["''", "Sentence", '"', "one", "."],
            ["Another", "\u201c", "WEIRD", "sentence", "!"],
        ]
        expected_spans = [
            [(0, 2), (2, 10), (10, 11), (13, 16), (16, 17)],
            [(0, 7), (8, 9), (9, 14), (15, 23), (25, 26)],
        ]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # sentence terminal tokenization
        text = "This is a ?sentence,. This is another sentence?,"
        expected_tokens = [
            ["This", "is", "a", "?", "sentence", ",", "."],
            ["This", "is", "another", "sentence", "?", ","],
        ]
        expected_spans = [
            [(0, 4), (5, 7), (8, 9), (10, 11), (11, 19), (19, 20), (20, 21)],
            [(0, 4), (5, 7), (8, 15), (16, 24), (24, 25), (25, 26)],
        ]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        #  Test random html escapes
        text = "&amp; sentence"
        expected_tokens = [["&", "amp", ";", "sentence"]]
        expected_spans = [[(0, 1), (1, 4), (4, 5), (6, 14)]]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # test that uris are properly split
        text = "http://www.foo.com/&amp;bar.html plus a sentence"
        expected_tokens = [["http://www.foo.com/&amp;bar.html", "plus", "a", "sentence"]]
        expected_spans = [[(0, 32), (33, 37), (38, 39), (40, 48)]]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # test possessive splitting
        text = "aron's duck"
        expected_tokens = [["aron", "'s", "duck"]]
        expected_spans = [[(0, 4), (4, 6), (7, 11)]]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        text = "kittens' delight"
        expected_tokens = [["kittens", "'", "delight"]]
        expected_spans = [[(0, 7), (7, 8), (9, 16)]]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # test contraction splitting
        text = "kittens you'd think wouldn't do that!"
        expected_tokens = [["kittens", "you", "'d", "think", "would", "n't", "do", "that", "!"]]
        expected_spans = [
            [(0, 7), (8, 11), (11, 13), (14, 19), (20, 25), (25, 28), (29, 31), (32, 36), (36, 37)]
        ]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # Test tokenize() with max_characters_per_token
        tokenizer = RegexTokenizer(max_characters_per_token=5)
        text = "kittens you'd think wouldn't do that!"
        expected_tokens = [["kitte", "ns", "you", "'d", "think", "would", "n't", "do", "that", "!"]]
        expected_spans = [
            [
                (0, 5),
                (5, 7),
                (8, 11),
                (11, 13),
                (14, 19),
                (20, 25),
                (25, 28),
                (29, 31),
                (32, 36),
                (36, 37),
            ]
        ]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        expected_tokens = [["kitte", "ns", "you", "'d", "think", "would", "n't", "do", "that", "!"]]
        expected_spans = [
            [
                (0, 5),
                (5, 7),
                (8, 11),
                (11, 13),
                (14, 19),
                (20, 25),
                (25, 28),
                (29, 31),
                (32, 36),
                (36, 37),
            ]
        ]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # extreme case of max_characters_per_token = 1
        tokenizer = RegexTokenizer(max_characters_per_token=1)
        text = "kittens you'd think wouldn't do that!"
        expected_tokens = [
            [
                "k",
                "i",
                "t",
                "t",
                "e",
                "n",
                "s",
                "y",
                "o",
                "u",
                "'",
                "d",
                "t",
                "h",
                "i",
                "n",
                "k",
                "w",
                "o",
                "u",
                "l",
                "d",
                "n",
                "'",
                "t",
                "d",
                "o",
                "t",
                "h",
                "a",
                "t",
                "!",
            ]
        ]
        expected_spans = [
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 28),
                (29, 30),
                (30, 31),
                (32, 33),
                (33, 34),
                (34, 35),
                (35, 36),
                (36, 37),
            ]
        ]
        tokens, spans = tokenizer.tokenize(text)
        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_spans, spans)

        # Test that an error is thrown if max_characters_per_token is not an integer
        with self.assertRaises(ValueError):
            tokenizer = RegexTokenizer(max_characters_per_token=0.1)

        # Test than an error is thrown if max_characters_per_token is less than 1
        with self.assertRaises(ValueError):
            tokenizer = RegexTokenizer(max_characters_per_token=0)

        # Test than an error is thrown if max_characters_per_token is negative
        with self.assertRaises(ValueError):
            tokenizer = RegexTokenizer(max_characters_per_token=-1)
