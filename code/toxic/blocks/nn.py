import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel


LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 6 * LSTM_UNITS


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, output_aux_sub=11):
        super().__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(
            embed_size, LSTM_UNITS, bidirectional=True, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True
        )

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS + 6 + output_aux_sub, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, 6)
        self.linear_sub_out = nn.Linear(DENSE_HIDDEN_UNITS, output_aux_sub)

    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        avg_pool1 = torch.mean(h_lstm1, 1)
        avg_pool2 = torch.mean(h_lstm2, 1)
        max_pool2, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((avg_pool1, max_pool2, avg_pool2), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        aux_result = self.linear_aux_out(hidden)
        sub_result = self.linear_sub_out(hidden)
        result = self.linear_out(torch.cat((hidden, aux_result, sub_result), 1))
        out = torch.cat([result, aux_result, sub_result], 1)
        return out


class GPT2CNN(GPT2PreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.cnn1 = nn.Conv1d(768, 256, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(256, num_labels, kernel_size=3, padding=1)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        lm_labels=None,
        past=None,
    ):
        x, _ = self.transformer(input_ids, position_ids, token_type_ids, past)
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn1(x))
        x = self.cnn2(x)
        output, _ = torch.max(x, 2)
        return output
