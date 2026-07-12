from torch import nn

import config


class ReviewAnalyzeModel(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=padding_index)
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_SIZE,batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_SIZE, 1)

    def forward(self, x):
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]
        output, _ = self.lstm(embed)
        # output.shape: [batch_size, seq_len, hidden_size]
        last_hidden = output[:, -1, :]
        # last_hidden.shape: [batch_size, hidden_size]
        output = self.linear(last_hidden)
        # output.shape: [batch_size, 1]
        return output
