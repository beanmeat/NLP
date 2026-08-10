from torch import nn
import config
from transformers import AutoModel


class ReviewAnalyzeModel(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(str(config.PRETRAINED_MODELS_DIR / 'bert-base-chinese'))
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # input_ids: [batch_size,seq_len]
        # attention_mask: [batch_size,seq_len]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # last_hidden_state.shape: [batch_size,seq_len,hidden_size]
        cls_output = last_hidden_state[:, 0, :]
        # cls_output.shape: [batch_size,hidden_size]

        output = self.linear(cls_output).squeeze(1)
        # output.shape: [batch_size]
        return output