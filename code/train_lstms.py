import os
import time
import random
import logging
import multiprocessing

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

import torch
from torch import nn
from torch.utils import data

from nltk import TweetTokenizer

from fastai.train import Learner
from fastai.train import DataBunch
from fastai.callbacks import OneCycleScheduler

from toxic.utils import seed_everything
from toxic.preprocessing import normalize
from toxic.blocks.data import SequenceBucketCollator
from toxic.blocks.nn import NeuralNet
from toxic.embeddings import gensim_to_embedding_matrix, one_hot_char_embeddings
from toxic.metrics import IDENTITY_COLUMNS


BATCH_SIZE = 320
EPOCHS = 5


def custom_loss(data, targets):
    bce_loss_1 = nn.BCEWithLogitsLoss(targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:7], targets[:, 2:8])
    bce_loss_3 = nn.BCEWithLogitsLoss(targets[:, 19:20])(
        data[:, 7:18], targets[:, 8:19]
    )
    return bce_loss_1 + bce_loss_2 + bce_loss_3 / 4


def save_nn_without_embedding_weights(model, path: str):
    temp_dict = model.state_dict()
    del temp_dict["embedding.weight"]
    torch.save(temp_dict, path)


if __name__ == "__main__":
    seed_everything(1234)
    torch.cuda.set_device(0)

    logging.info("Reading data...")
    INPUT_FOLDER = "../input/jigsaw-unintended-bias-in-toxicity-classification/"
    train = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    y = train["target"].values

    logging.info("Preprocessing...")
    with multiprocessing.Pool(processes=32) as pool:
        text_list = pool.map(normalize, train.comment_text.tolist())

    logging.info("Tokenization...")
    tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    word_sequences = []
    word_dict = {}
    word_counter = {}
    word_index = 1

    for doc in tqdm(text_list):
        word_seq = []
        for token in tweet_tokenizer.tokenize(doc):
            if token not in word_dict:
                word_dict[token] = word_index
                word_counter[token] = 1
                word_index += 1
            word_seq.append(word_dict[token])
            word_counter[token] = +1
        word_sequences.append(word_seq)

    lengths = torch.from_numpy(np.array([len(x) for x in word_sequences]))
    maxlen = lengths.max()

    x_train_padded = torch.tensor(pad_sequences(word_sequences, maxlen=maxlen)).long()

    train_collator = SequenceBucketCollator(
        lambda lenghts: lenghts.max(),
        maxlen=maxlen,
        sequence_index=0,
        length_index=1,
        label_index=2,
    )

    logging.info("Loading pretrained embeddings...")

    glove_matrix, _ = gensim_to_embedding_matrix(
        word_dict, "../gensim-embeddings-dataset/glove.840B.300d.gensim"
    )
    crawl_matrix, _ = gensim_to_embedding_matrix(
        word_dict, "../gensim-embeddings-dataset/crawl-300d-2M.gensim"
    )
    para_matrix, _ = gensim_to_embedding_matrix(
        word_dict, "../gensim-embeddings-dataset/paragram_300_sl999.gensim"
    )
    w2v_matrix, _ = gensim_to_embedding_matrix(
        word_dict, "../gensim-embeddings-dataset/GoogleNews-vectors-negative300.gensim"
    )

    logging.info("Buliding char matrix...")
    vectorizer = CountVectorizer(analyzer="char")
    char_matrix = one_hot_char_embeddings(word_dict, vectorizer)

    logging.info("Buliding targets and weights ...")
    iden = train[IDENTITY_COLUMNS].fillna(0).values
    subgroup_target = np.hstack(
        [
            (iden >= 0.5).any(axis=1, keepdims=True).astype(np.int),
            iden,
            iden.max(axis=1, keepdims=True),
        ]
    )
    sub_target_weigths = (
        ~train[IDENTITY_COLUMNS].isna().values.any(axis=1, keepdims=True)
    ).astype(np.int)

    train["is_idententy"] = (iden >= 0.5).any().astype(np.int)

    weights = np.ones(len(train))
    weights += (iden >= 0.5).any(1)
    weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
    weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
    weights += (
        (
            (train["target"].values >= 0.5).astype(bool).astype(np.int)
            + (train["is_idententy"].values == 1)
        )
        .astype(bool)
        .astype(np.int)
    )

    weights /= weights.mean()

    AUX_TARGETS = [
        "target",
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
    ]

    y_aux_train = train[AUX_TARGETS]

    def get_y_train_torch(weights):
        return torch.tensor(
            np.hstack(
                [
                    y[:, None],
                    weights[:, None],
                    y_aux_train,
                    subgroup_target,
                    sub_target_weigths,
                ]
            )
        ).float()

    def get_databunch(y_train_torch):
        train_dataset = data.TensorDataset(x_train_padded, lengths, y_train_torch)
        # Some tricky stuff to make DataBunch work
        valid_dataset = data.TensorDataset(
            x_train_padded[:BATCH_SIZE],
            lengths[:BATCH_SIZE],
            y_train_torch[:BATCH_SIZE],
        )
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=train_collator,
        )
        valid_loader = data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=train_collator,
        )
        return DataBunch(
            train_dl=train_loader, valid_dl=valid_loader, collate_fn=train_collator
        )

    y_train_torch = get_y_train_torch(weights)
    databunch = get_databunch(y_train_torch)

    logging.info("training model 1: para, rawl, w2v...")
    embedding_matrix = np.concatenate(
        [para_matrix, crawl_matrix, w2v_matrix, char_matrix], axis=1
    )
    seed_everything(42)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_1.bin")

    logging.info("training model 2: glove, crawl, w2v...")
    embedding_matrix = np.concatenate(
        [glove_matrix, crawl_matrix, w2v_matrix, char_matrix], axis=1
    )
    seed_everything(43)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_2.bin")

    logging.info("training model 3: glove, para, w2v...")
    embedding_matrix = np.concatenate(
        [glove_matrix, para_matrix, w2v_matrix, char_matrix], axis=1
    )
    seed_everything(44)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_3.bin")

    logging.info("training model 4: glove, para, crawl...")
    embedding_matrix = np.concatenate(
        [glove_matrix, para_matrix, crawl_matrix, char_matrix], axis=1
    )
    seed_everything(45)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_4.bin")

    logging.info("training model 5: base ...")
    embedding_matrix = np.concatenate(
        [glove_matrix, crawl_matrix, w2v_matrix, char_matrix], axis=1
    )
    seed_everything(46)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_5.bin")

    logging.info("training model 6: base + homosexual boost ...")
    weights = np.ones(len(train))
    weights += (iden >= 0.5).any(1)
    weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
    weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
    weights += (
        (
            (train["target"].values >= 0.5).astype(bool).astype(np.int)
            + (train["is_idententy"].values == 1)
        )
        .astype(bool)
        .astype(np.int)
    )
    weights += (
        (train[["homosexual_gay_or_lesbian"]].fillna(0).values >= 0.5)
        .sum(axis=1)
        .astype(bool)
        .astype(np.int)
    )
    weights /= weights.mean()
    y_train_torch = get_y_train_torch(weights)
    databunch = get_databunch(y_train_torch)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_6.bin")

    logging.info("training model 7: base + psychiatric boost ...")
    weights = np.ones(len(train))
    weights += (iden >= 0.5).any(1)
    weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
    weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
    weights += (
        (
            (train["target"].values >= 0.5).astype(bool).astype(np.int)
            + (train["is_idententy"].values == 1)
        )
        .astype(bool)
        .astype(np.int)
    )
    weights += (
        (train[["psychiatric_or_mental_illness"]].fillna(0).values >= 0.5)
        .sum(axis=1)
        .astype(bool)
        .astype(np.int)
    )
    weights /= weights.mean()
    y_train_torch = get_y_train_torch(weights)
    databunch = get_databunch(y_train_torch)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_7.bin")

    logging.info("training model 8: base + jewish boost ...")
    weights = np.ones(len(train))
    weights += (iden >= 0.5).any(1)
    weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
    weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
    weights += (
        (
            (train["target"].values >= 0.5).astype(bool).astype(np.int)
            + (train["is_idententy"].values == 1)
        )
        .astype(bool)
        .astype(np.int)
    )
    weights += (
        (train[["psychiatric_or_mental_illness"]].fillna(0).values >= 0.5)
        .sum(axis=1)
        .astype(bool)
        .astype(np.int)
    )
    weights /= weights.mean()
    y_train_torch = get_y_train_torch(weights)
    databunch = get_databunch(y_train_torch)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_8.bin")

    logging.info("training model 9: base + black boost ...")
    weights = np.ones(len(train))
    weights += (iden >= 0.5).any(1)
    weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
    weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
    weights += (
        (
            (train["target"].values >= 0.5).astype(bool).astype(np.int)
            + (train["is_idententy"].values == 1)
        )
        .astype(bool)
        .astype(np.int)
    )
    weights += (
        (train[["psychiatric_or_mental_illness"]].fillna(0).values >= 0.5)
        .sum(axis=1)
        .astype(bool)
        .astype(np.int)
    )
    weights /= weights.mean()
    y_train_torch = get_y_train_torch(weights)
    databunch = get_databunch(y_train_torch)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_9.bin")

    logging.info("training model 10: base + white boost ...")
    weights = np.ones(len(train))
    weights += (iden >= 0.5).any(1)
    weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
    weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
    weights += (
        (
            (train["target"].values >= 0.5).astype(bool).astype(np.int)
            + (train["is_idententy"].values == 1)
        )
        .astype(bool)
        .astype(np.int)
    )
    weights += (
        (train[["psychiatric_or_mental_illness"]].fillna(0).values >= 0.5)
        .sum(axis=1)
        .astype(bool)
        .astype(np.int)
    )
    weights /= weights.mean()
    y_train_torch = get_y_train_torch(weights)
    databunch = get_databunch(y_train_torch)
    model = NeuralNet(embedding_matrix, output_aux_sub=subgroup_target.shape[1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    cb = OneCycleScheduler(learn, lr_max=0.001)
    learn.callbacks.append(cb)
    learn.fit(EPOCHS)
    save_nn_without_embedding_weights(learn.model, "./models/Notebook_100_10.bin")
