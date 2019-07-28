import os
import logging
import multiprocessing

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils import data

from apex import amp
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam

from toxic.utils import (
    perfect_bias,
    TensorboardAggregator,
    clip_to_max_len,
    should_decay,
)
from toxic.metrics import IDENTITY_COLUMNS
from toxic.bert import PipeLineConfig, convert_line_gpt, AUX_TARGETS
from toxic.utils import seed_everything, convert_dataframe_to_bool
from toxic.blocks.nn import GPT2CNN

BATCH_SIZE = 64
ACCUM_STEPS = 2


def train_gpt(config: PipeLineConfig):
    logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")
    input_folder = "../input/jigsaw-unintended-bias-in-toxicity-classification/"
    train = pd.read_csv(os.path.join(input_folder, "train.csv"))

    logging.info("Reading wiki PL...")
    wiki_sents = pd.read_csv("../input/wiki_sents.csv")
    wiki_subset = wiki_sents[
        (wiki_sents.target < 0.1) & (wiki_sents[IDENTITY_COLUMNS].max(1) >= 0.33)
    ].copy()
    wiki_subset.drop(
        columns=["any_identity", "max_identity", "target_aux"], inplace=True
    )
    wiki_subset.iloc[:, :6] = 0.0  # They are not toxic by definition

    logging.info("Sampling extra data...")
    seed_everything(config.seed + 1)
    extras = []
    t = convert_dataframe_to_bool(train)
    for identity in IDENTITY_COLUMNS:
        Ip = np.sum(t[identity] & t.target)
        I = np.sum(t[identity])
        Bp = np.sum(~t[identity] & t.target)
        B = np.sum(~t[identity])
        required = (Ip * B - Bp * I) // Bp

        extra = wiki_subset[wiki_subset[identity] >= 0.333].copy()
        logging.info("Mitigating bias for %s", identity)
        logging.info("Need %d extra samples, got %d", required, len(extra))
        if len(extra) > required:
            logging.info("Downsampling extra dataframe")
            extra = extra.sample(required)
        extras.append(extra)

    enriched = pd.concat([train] + extras, ignore_index=True, sort=False, axis=0)

    logging.info("Tokenizing...")

    with multiprocessing.Pool(processes=32) as pool:
        text_list = enriched.comment_text.tolist()
        sequences = pool.map(convert_line_gpt, text_list)

    logging.info("Building ttensors for training...")
    sequences = np.array(sequences)
    print(sequences.shape)
    lengths = np.argmax(sequences == 0, axis=1)
    lengths[lengths == 0] = sequences.shape[1]

    logging.info("Bulding target tesnor...")
    iden = enriched[IDENTITY_COLUMNS].fillna(0).values
    subgroup_target = np.hstack(
        [
            (iden >= 0.5).any(axis=1, keepdims=True).astype(np.int),
            iden,
            iden.max(axis=1, keepdims=True),
        ]
    )
    sub_target_weigths = (
        ~enriched[IDENTITY_COLUMNS].isna().values.any(axis=1, keepdims=True)
    ).astype(np.int)

    weights = np.ones(len(enriched))
    weights += (iden >= 0.5).any(1)
    weights += (enriched["target"].values >= 0.5) & (iden < 0.5).any(1)
    weights += (enriched["target"].values < 0.5) & (iden >= 0.5).any(1)
    weights /= weights.mean()

    y_aux_train = enriched[AUX_TARGETS]
    y_train_torch = torch.tensor(
        np.hstack(
            [
                enriched.target.values[:, None],
                weights[:, None],
                y_aux_train,
                subgroup_target,
                sub_target_weigths,
            ]
        )
    ).float()

    logging.info("Seeding with seed %d ...", config.seed)
    seed_everything(config.seed)

    logging.info("Creating dataset...")
    dataset = data.TensorDataset(
        torch.tensor(sequences), y_train_torch, torch.tensor(lengths)
    )
    train_loader = data.DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=clip_to_max_len, shuffle=True
    )

    logging.info("Creating a model...")
    model = GPT2CNN.from_pretrained("gpt2", num_labels=18)
    model.zero_grad()
    model = model.cuda()

    logs_file = f"./tb_logs/final_{config.expname}"
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if should_decay(n)],
            "weight_decay": config.decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if not should_decay(n)],
            "weight_decay": 0.00,
        },
    ]

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=config.lr,
        warmup=config.warmup,
        t_total=config.epochs * len(train_loader) // ACCUM_STEPS,
    )

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    model = model.train()

    writer = SummaryWriter(logs_file)
    agg = TensorboardAggregator(writer)
    custom_loss = prepare_loss(config)

    for _ in range(config.epochs):
        for j, (X, y) in enumerate(train_loader):

            X = X.cuda()
            y = y.cuda()

            y_pred = model(X)
            loss = custom_loss(y_pred, y)

            accuracy = ((y_pred[:, 0] > 0) == (y[:, 0] > 0.5)).float().mean()
            agg.log({"train_loss": loss.item(), "train_accuracy": accuracy.item()})

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (j + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

    torch.save(model.state_dict(), f"./models/final-pipe6-{config.expname}.bin")


if __name__ == "__main__":
    config_1 = PipeLineConfig(
        expname="gpt_wiki_1",
        lr=4.9e-5,
        warmup=0.06,
        epochs=2,
        seed=50462,
        decay=0.04,
        main_loss_weight=1.05,
    )
    config_2 = PipeLineConfig(
        expname="gpt_wiki_2",
        lr=4.7e-5,
        warmup=0.055,
        epochs=2,
        seed=54184,
        decay=0.06,
        main_loss_weight=0.98,
    )

    for config in (config_1, config_2):
        train_gpt(config)
