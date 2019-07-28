from torch import nn
from pytorch_pretrained_bert import BertTokenizer, GPT2Tokenizer


UNCASED_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
CASED_TOKENIZER = BertTokenizer.from_pretrained("bert-base-cased")
GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
MAX_LEN = 300 - 2
GPT_MAX_LEN = 250

AUX_TARGETS = [
    "target",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
]


class PipeLineConfig:
    def __init__(self, lr, warmup, epochs, seed, expname, decay, main_loss_weight):
        self.lr = lr
        self.warmup = warmup
        self.epochs = epochs
        self.seed = seed
        self.expname = expname
        self.decay = decay
        self.main_loss_weight = main_loss_weight


def convert_line_uncased(text):
    tokens_a = UNCASED_TOKENIZER.tokenize(text)[:MAX_LEN]
    one_token = UNCASED_TOKENIZER.convert_tokens_to_ids(
        ["[CLS]"] + tokens_a + ["[SEP]"]
    )
    one_token += [0] * (MAX_LEN - len(tokens_a))
    return one_token


def convert_line_cased(text):
    tokens_a = CASED_TOKENIZER.tokenize(text)[:MAX_LEN]
    one_token = CASED_TOKENIZER.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"])
    one_token += [0] * (MAX_LEN - len(tokens_a))
    return one_token


def convert_line_gpt(text):
    tokens_a = GPT2_TOKENIZER.tokenize(text)[:GPT_MAX_LEN]
    one_token = GPT2_TOKENIZER.convert_tokens_to_ids(tokens_a)
    one_token += [0] * (GPT_MAX_LEN - len(tokens_a))
    return one_token


def prepare_loss(config: PipeLineConfig):
    def custom_loss(data, targets):
        bce_loss_1 = nn.BCEWithLogitsLoss(targets[:, 1:2])(data[:, :1], targets[:, :1])
        bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:7], targets[:, 2:8])
        bce_loss_3 = nn.BCEWithLogitsLoss(targets[:, 19:20])(
            data[:, 7:18], targets[:, 8:19]
        )
        return config.main_loss_weight * bce_loss_1 + bce_loss_2 + bce_loss_3 / 4

    return custom_loss
