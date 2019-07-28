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
from toxic.bert import convert_line_uncased, PipeLineConfig, AUX_TARGETS
from toxic.utils import seed_everything

BATCH_SIZE = 64
ACCUM_STEPS = 2


def train_bert(config: PipeLineConfig):
    logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")
    input_folder = "../input/jigsaw-unintended-bias-in-toxicity-classification/"
    train = pd.read_csv(os.path.join(input_folder, "train.csv"))

    logging.info("Tokenizing...")

    with multiprocessing.Pool(processes=32) as pool:
        text_list = train.comment_text.tolist()
        sequences = pool.map(convert_line_uncased, text_list)

    logging.info("Building ttensors for training...")
    sequences = np.array(sequences)
    lengths = np.argmax(sequences == 0, axis=1)
    lengths[lengths == 0] = sequences.shape[1]

    logging.info("Bulding target tesnor...")
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

    weights = np.ones(len(train))
    weights += (iden >= 0.5).any(1)
    weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
    weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
    weights /= weights.mean()

    y_aux_train = train[AUX_TARGETS]
    y_train_torch = torch.tensor(
        np.hstack(
            [
                train.target.values[:, None],
                weights[:, None],
                y_aux_train,
                subgroup_target,
                sub_target_weigths,
            ]
        )
    ).float()

    perfect_output = torch.tensor(
        np.hstack([train.target.values[:, None], y_aux_train, subgroup_target])
    ).float()

    logging.info("Seeding with seed %d ...", config.seed)
    seed_everything(config.seed)

    logging.info("Creating dataset...")
    dataset = data.TensorDataset(
        torch.from_numpy(sequences).long(), y_train_torch, torch.from_numpy(lengths)
    )
    train_loader = data.DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=clip_to_max_len, shuffle=True
    )

    logging.info("Creating a model...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=18
    )
    model.zero_grad()
    model = model.cuda()
    model.classifier.bias = nn.Parameter(perfect_bias(perfect_output.mean(0)).cuda())

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

            y_pred = model(X, attention_mask=(X > 0))
            loss = custom_loss(y_pred, y)

            accuracy = ((y_pred[:, 0] > 0) == (y[:, 0] > 0.5)).float().mean()
            agg.log({"train_loss": loss.item(), "train_accuracy": accuracy.item()})

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (j + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

    torch.save(model.state_dict(), f"./models/final-pipe2-{config.expname}.bin")


if __name__ == "__main__":
    config_1 = PipeLineConfig(
        expname="raw_1",
        lr=4e-5,
        warmup=0.05,
        epochs=2,
        seed=1992,
        decay=0.05,
        main_loss_weight=1.0,
    )
    config_2 = PipeLineConfig(
        expname="raw_2",
        lr=4.5e-5,
        warmup=0.06,
        epochs=2,
        seed=905,
        decay=0.05,
        main_loss_weight=1.11,
    )
    config_3 = PipeLineConfig(
        expname="raw_3",
        lr=4e-5,
        warmup=0.05,
        epochs=2,
        seed=130000,
        decay=0.055,
        main_loss_weight=0.95,
    )

    for config in (config_1, config_2, config_3):
        train_bert(config)
