import fastText
import numpy as np
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

PORTER_STEMMER = PorterStemmer()
LANCASTER_STEMMER = LancasterStemmer()
SNOWBALL_STEMMER = SnowballStemmer("english")


WORD_TRANSFORMERS = [
    lambda x: x.lower(),
    lambda x: x.upper(),
    lambda x: x.capitalize(),
    PORTER_STEMMER.stem,
    LANCASTER_STEMMER.stem,
    SNOWBALL_STEMMER.stem,
]


def word_forms(word):
    """
        Yields different forms of word, one by one
    """
    yield word
    yield from (transformer(word) for transformer in WORD_TRANSFORMERS)


def maybe_get_embedding(word, model):
    for form in word_forms(word):
        if form in model:
            return model[form]

    word = word.strip("-'")
    for form in word_forms(word):
        if form in model:
            return model[form]

    return None


def construct_empty(word2index, dim):
    return np.zeros((max(word2index.values()) + 1, dim), dtype=np.float32)


def gensim_to_embedding_matrix(word2index, path):
    """
        path - path to model obtained with gensim.models.KeyedVectors.save
    """
    model = KeyedVectors.load(path)
    embedding_matrix = construct_empty(word2index, model.vector_size)
    unknown_words = []

    for word, i in word2index.items():
        maybe_embedding = maybe_get_embedding(word, model)
        if maybe_embedding is not None:
            embedding_matrix[i] = maybe_embedding
        else:
            unknown_words.append(word)

    return embedding_matrix, unknown_words


def fasttext_to_embedding_matrix(word2index, path):
    """
        path - path to binary model obtained from fasttext util
    """
    model = fastText.load_model(path)
    embedding_matrix = construct_empty(word2index, model.get_dimension())
    for word, index in word2index.items():
        embedding_matrix[index] = model.get_word_vector(word)
    return embedding_matrix


def one_hot_char_embeddings(word2index, char_vectorizer):
    """
        vectorizer - fitted vectorizer
    """

    words = [""] * (max(word2index.values()) + 1)
    for word, i in word2index.items():
        words[i] = word

    return char_vectorizer.transform(words).toarray().astype(np.float32)
