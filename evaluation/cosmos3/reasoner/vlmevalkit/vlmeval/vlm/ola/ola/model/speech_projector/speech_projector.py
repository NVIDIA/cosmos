import math

import torch
import torch.nn as nn


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.hidden_size)

        embed_std = 1 / math.sqrt(config.hidden_size)
        self.speech_newline = nn.Parameter(
            torch.randn(config.hidden_size) * embed_std
        )
        self.speech_begin = nn.Parameter(
            torch.randn(config.hidden_size) * embed_std
        )
        self.speech_end = nn.Parameter(
            torch.randn(config.hidden_size) * embed_std
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = torch.cat([
            x,
            self.speech_newline.reshape(1, 1, -1).expand(batch_size, 1, -1).to(x.dtype)
        ], dim=1)
        begin = self.speech_begin.reshape(1, -1).to(x.dtype)
        end = self.speech_end.reshape(1, -1).to(x.dtype)
        x = x.flatten(0, 1)
        x = torch.cat([begin, x, end], dim=0)
        # x = x.flatten(0, 1)
        return x
