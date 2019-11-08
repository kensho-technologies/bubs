# Copyright 2019 Kensho Technologies, LLC.
"""Fake tests that attempt to at least import a few files."""
import unittest


class TestImports(unittest.TestCase):
    def test_imports(self):
        from ..embedding_layer import ContextualizedEmbedding  # noqa
        from ..helpers import InputEncoder  # noqa
