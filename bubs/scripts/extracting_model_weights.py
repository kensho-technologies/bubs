# Copyright 2019 Kensho Technologies, LLC.
"""For your reference only: not required for using bubs.

This is a script that shows how model weights were extracted from flair embeddings.
It requires flair: pip install flair before running this.
"""

import numpy as np

from flair.embeddings import FlairEmbeddings


def extract_weights():
    """Copy the exact weights in flair's news-forward-fast and news-backward-fast embeddings."""
    flair_forward_embeddings = FlairEmbeddings("news-forward-fast")
    flair_backward_embeddings = FlairEmbeddings("news-backward-fast")

    """These are the matrices we provide in the flair_news_fast_weights.npz file"""
    char_embed_weights_forward = (
        flair_forward_embeddings.lm.encoder._parameters["weight"].detach().cpu().numpy()
    )
    char_embed_weights_back = (
        flair_backward_embeddings.lm.encoder._parameters["weight"].detach().cpu().numpy()
    )

    weight_input_to_hidden_forward = (
        flair_forward_embeddings.lm.rnn._parameters["weight_ih_l0"].detach().cpu().numpy()
    )
    weight_hidden_to_hidden_forward = (
        flair_forward_embeddings.lm.rnn._parameters["weight_hh_l0"].detach().cpu().numpy()
    )
    bias_input_to_hidden_forward = (
        flair_forward_embeddings.lm.rnn._parameters["bias_ih_l0"].detach().cpu().numpy()
    )
    bias_hidden_to_hidden_forward = (
        flair_forward_embeddings.lm.rnn._parameters["bias_hh_l0"].detach().cpu().numpy()
    )
    weight_input_to_hidden_back = (
        flair_backward_embeddings.lm.rnn._parameters["weight_ih_l0"].detach().cpu().numpy()
    )
    weight_hidden_to_hidden_back = (
        flair_backward_embeddings.lm.rnn._parameters["weight_hh_l0"].detach().cpu().numpy()
    )
    bias_input_to_hidden_back = (
        flair_backward_embeddings.lm.rnn._parameters["bias_ih_l0"].detach().cpu().numpy()
    )
    bias_hidden_to_hidden_back = (
        flair_backward_embeddings.lm.rnn._parameters["bias_hh_l0"].detach().cpu().numpy()
    )

    filename = "flair_news_fast_weights.npz"
    np.savez_compressed(
        filename,
        weight_input_to_hidden_forward=weight_input_to_hidden_forward,
        weight_hidden_to_hidden_forward=weight_hidden_to_hidden_forward,
        bias_input_to_hidden_forward=bias_input_to_hidden_forward,
        bias_hidden_to_hidden_forward=bias_hidden_to_hidden_forward,
        weight_input_to_hidden_back=weight_input_to_hidden_back,
        weight_hidden_to_hidden_back=weight_hidden_to_hidden_back,
        bias_input_to_hidden_back=bias_input_to_hidden_back,
        bias_hidden_to_hidden_back=bias_hidden_to_hidden_back,
        char_embed_weights_forward=char_embed_weights_forward,
        char_embed_weights_back=char_embed_weights_back,
    )


if __name__ == "__main__":
    extract_weights()
