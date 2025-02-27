"""Construct a BagOfWords component."""

from dataclasses import dataclass
from pathlib import Path

import ground_texture_slam


@dataclass
class BagOfWordsOptions:
    """Options for the BagOfWords component."""

    vocab_file: Path


def create_bag_of_words(
    options: BagOfWordsOptions,
) -> ground_texture_slam.BagOfWords:
    """Create a BagOfWords object with a predefined vocabulary.

    Parameters
    ----------
    options
        Options for this component.

    Returns
    -------
        The constructed object.
    """
    bag_of_words_options = ground_texture_slam.BagOfWords.Options()
    bag_of_words_options.vocab_file = options.vocab_file.absolute().as_posix()
    return ground_texture_slam.BagOfWords(bag_of_words_options)
