from tensorboard import summary
from torch import nn
import torch
import config


# 编码器
class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.ENCODER_HIDDEN_SIZE,
                          batch_first=True,
                          num_layers=config.ENCODER_LAYERS,
                          bidirectional=True)

    def forward(self, x):
        # x.shape: [batch_size,seq_len]
        embed = self.embedding(x)
        # embed.shape: [batch_size,seq_len,embedding_dim]
        output, hidden = self.gru(embed)
        # hidden.shape: [num_layer * direction,batch_size,hidden_size]
        last_hidden_forward = hidden[-2]
        last_hidden_backward = hidden[-1]
        context_vector = torch.cat([last_hidden_forward, last_hidden_backward], dim=1)
        # context_vector.shape: [batch_size,hidden_size * 2]
        return context_vector


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_index
        )
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.DECODER_HIDDEN_SIZE,
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=config.DECODER_HIDDEN_SIZE,
            out_features=vocab_size
        )

    def forward(self, tgt, hidden):
        embedded = self.embedding(tgt)  # (batch_size, 1, embedding_dim)
        output, hidden = self.gru(embedded, hidden)  # output: (batch_size, 1, hidden_dim)
        output = self.linear(output)  # (batch_size, 1, vocab_size)
        return output, hidden
