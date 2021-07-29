# Copyright 2019 Kensho Technologies, LLC.
"""For your reference only: not required for using bubs.

This is a script that shows how model weights and char-to-int map were extracted from flair embeddings.
It requires flair: pip install flair before running this.
"""

import numpy as np
import pickle

from flair.embeddings import FlairEmbeddings


def extract_weights_and_charmap(fw_name="news-forward", bw_name="news-backward", weights_name="news_weights.npz", charmap_name="news_charmap.pkl"):
    """Copy the exact weights in the given flair's forward and backward embeddings into numpy compressed file
    and copy the exact char-to-int map into dictionary pickle."""

    flair_forward_embeddings = FlairEmbeddings(fw_name)
    flair_backward_embeddings = FlairEmbeddings(bw_name)

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

    np.savez_compressed(
        weights_name,
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

    """This extracts char-to-int map into dictionary pickle."""
    with open(charmap_name, "wb") as f:
        pickle.dump(flair_forward_embeddings.lm.dictionary.item2idx, f)


if __name__ == "__main__":
    """Example list of tuples of English Flair embedding names (forward and backward) and extracted weights and charmap file names."""
    NAMES = [
        ("news-forward", "news-backward", "news_weights.npz", "news_charmap.pkl"),
        ("news-forward-fast", "news-backward-fast", "news_fast_weights.npz", "news_fast_charmap.pkl"),
        ("mix-forward", "mix-backward", "mix_weights.npz", "mix_charmap.pkl"),
    ]
    fw_name, bw_name, weights_name, charmap_name = NAMES[1]

    extract_weights_and_charmap(fw_name=fw_name, bw_name=bw_name, weights_name=weights_name, charmap_name=charmap_name)
