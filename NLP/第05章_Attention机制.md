## 第05章_Attention机制

------

### 5.1 概述

传统的 Seq2Seq 模型中，编码器在处理源句时，无论其长度如何，最终都只能将整句信息压缩为一个固定长度的上下文向量，用作解码器的唯一参考。这种设计存在两个显著问题：

- 信息压缩困难：**固定向量难以完整表达长句或复杂语义**，容易丢失关键信息；
- 缺乏动态感知：解码器在每一步生成中都只能依赖同一个上下文向量，难以根据不同位置的生成需要灵活提取信息；

为了解决上述问题，研究者引入了 Attention 机制。其核心思想是：

解码器在生成目标序列的每一步时，不再依赖于一个静态的上下文向量，而是根据当前的解码状态，动态地从编码器各时间步的隐藏状态中选取最相关的信息，以辅助当前步的生成。

这种机制赋予模型“对齐”能力，使其能够自动判断源句中哪些位置对当前的目标词更为重要，从而有效缓解信息瓶颈问题，提升生成质量与表达能力。

### 5.2 工作原理

注意力机制的核心思想，是解码器在生成目标序列的每一步时，动态地从编码器的各个时间步的隐藏状态中提取当前所需的信息，而不再只依赖一个固定的上下文向量。

![image-20260714152507809](images/image-20260714152507809.png)

这一机制通常通过以下 4 个关键步骤实现：

#### 5.2.1 相关性计算

在目标序列生成的每一步，解码器都会计算当前时间步的隐藏状态与编码器各个时间步输出之间的相关性。这些相关性衡量了源句中每个位置对当前生成内容的重要程度，从而决定模型应将多少注意力分配给不同的源位置。

相关性的计算依赖于特定的函数，通常被称为注意力评分函数（`attention scoring function`）。常见的评分函数实现方式将在下一节中详细介绍。

![image-20260714224848467](images/image-20260714224848467.png)

#### 5.2.2 注意力权重计算

得到所有源位置的注意力评分后，使用 Softmax 函数将其归一化为概率分布，作为注意力权重。得分越高的位置，其对应的权重越大，代表模型在当前生成中更关注该位置的信息。

![image-20260714224911717](images/image-20260714224911717.png)

#### 5.2.3 上下文向量计算

将所有编码器输出按照注意力权重进行加权求和，得到一个上下文向量。这个向量就表示当前时间步，模型从源句中提取出的关键信息。

![image-20260714224934200](images/image-20260714224934200.png)

#### 5.2.4 解码信息融合

在得到上下文向量后，解码器将其与当前时间步的隐藏状态进行拼接，以融合两者信息，最终通过线性变换和 Softmax，生成当前时间步目标词的概率分布。

![image-20260714224952362](images/image-20260714224952362.png)

### 5.3 注意力评分函数

#### 5.3.1 概述

注意力评分函数有多种实现方式。本节将介绍三种常见的计算方法：点积评分（Dot）、通用点积评分（General）和拼接评分（Concat）。它们虽然在结构上各有差异，但本质上都是用于衡量解码器当前隐藏状态与编码器各时间步隐藏状态之间的相关性，并据此分配注意力权重。

#### 5.3.2 点积评分（Dot）

点积评分是注意力机制中最简单、最直接的一种相关性评分方法。它通过计算解码器当前时间步的隐藏状态与编码器每个时间步的隐藏状态的点积，来衡量二者之间的相关性：

![image-20260714225017200](images/image-20260714225017200.png)

其含义可以理解为：如果两个向量方向越一致（即越接近），它们的点积就越大，表示相关性越强，模型应当给予更多注意力。

#### 5.3.3 通用点积评分（General）

通用点积评分在点积的基础上引入了一个可学习的权重矩阵W,用于先对编码器隐藏状态进行线性变换，再与解码器隐藏状态进行点积：

![image-20260714225038635](images/image-20260714225038635.png)

该方法的设计动机主要是为了解决**编码器和解码器隐藏状态维度不一致的问题**。通过引入权重矩阵W，不仅实现了维度对齐，也增强了模型对编码器输出的适应能力，从而提升了注意力机制的表达能力。

#### 5.3.4 拼接评分（Concat）

拼接评分是一种表达能力更强的相关性评分方法。它的核心思想是：将解码器当前隐藏状态与编码器每个时间步的隐藏状态拼接为一个长向量，经过线性变换和非线性激活，最后用一个向量进行投影，得到最终打分值：

![image-20260714225102323](images/image-20260714225102323.png)

相比前两种方法，Concat 评分方式在建模能力上更强。它不仅考虑了两个状态的数值关系，还引入非线性变换，能够捕捉更复杂的交互模式，更适合处理对齐关系复杂的任务场景。

### 5.4 案例实操（中英翻译V2.0）

#### 5.4.1 需求说明

本案例要求在已有的 Seq2Seq 模型基础上，引入注意力机制，以提升模型在处理长句或复杂句时的表达能力和生成质量。

#### 5.4.2 需求实现

本案例要求在已有的 Seq2Seq 模型基础上，引入注意力机制，以提升模型在处理长句或复杂句时的表达能力和生成质量。

- 编码器

  编码器无需任何改变；

- 解码器

  解码器在每个时间步，都需要将当前隐藏状态与编码器输出序列共同用于计算注意力权重（使用点积评分函数）；之后根据权重对编码器各位置进行加权求和，得到上下文向量；最后再将上下文向量与当前解码状态拼接，作为输出的最终依据。

#### 5.4.3 需求实现

1. 项目结构

   ![image-20260714225205254](images/image-20260714225205254.png)

2. 完整代码

   - 数据预处理

     ```python
     import pandas as pd
     from sklearn.model_selection import train_test_split
     from tokenizer import EnglishTokenizer, ChineseTokenizer
     
     import config
     
     
     def process():
         print('开始处理数据')
         # 读取数据
         df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt', sep='\t', header=None, usecols=[0, 1], encoding='utf-8',
                          names=["en", "zh"])
     
         # 过滤数据
         df = df.dropna()
         df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]
         # print(df.head())
     
         # 划分数据集
         train_df, test_df = train_test_split(df, test_size=0.2)
     
         # 构建词表
         ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.PROCESSED_DIR / 'zh_vocab.txt')
         EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.PROCESSED_DIR / 'en_vocab.txt')
     
         # 构建tokenizer对象
         zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR / 'zh_vocab.txt')
         en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR / 'en_vocab.txt')
     
         # 计算序列长度（95%分位数）
         # zh_len = train_df['zh'].apply(lambda x: len(zh_tokenizer.tokenize(x))).max()
         # en_len = train_df['en'].apply(lambda x: len(en_tokenizer.tokenize(x))).max()
         # print(zh_len,en_len)
     
         # 构建训练集
         train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=False))
         train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=True))
         # 保存训练集
         train_df.to_json(config.PROCESSED_DIR / 'indexed_train.jsonl', orient='records', lines=True)
         # 构建测试集
         test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=False))
         test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=True))
         # 保存测试集
         test_df.to_json(config.PROCESSED_DIR / 'indexed_test.jsonl', orient='records', lines=True)
     
         print('数据处理完成')
     
     
     if __name__ == '__main__':
         process()
     
     ```
   
   - 自定义分词器
   
     ```python
     from abc import abstractmethod
     
     import nltk
     from nltk import word_tokenize, TreebankWordDetokenizer
     from tqdm import tqdm
     
     
     class BaseTokenizer:
         unk_token = '<unk>'
         pad_token = '<pad>'
         sos_token = '<sos>'
         eos_token = '<eos>'
     
         def __init__(self, vocab_list):
             self.vocab_list = vocab_list
             self.vocab_size = len(vocab_list)
     
             self.word2index = {word: index for index, word in enumerate(vocab_list)}
             self.index2word = {index: word for index, word in enumerate(vocab_list)}
     
             self.unk_token_id = self.word2index.get(self.unk_token)
             self.pad_token_id = self.word2index.get(self.pad_token)
             self.sos_token_id = self.word2index.get(self.sos_token)
             self.eos_token_id = self.word2index.get(self.eos_token)
     
         @staticmethod
         @abstractmethod
         def tokenize(text):
             """
             分词抽象方法
             :param text: 文本
             :return:
             """
             pass
     
         @abstractmethod
         def decode(self, word_ids):
             """
             解码抽象方法
             :param word_ids: 索引
             :return: 字符串
             """
             pass
     
     
         def encode(self, text, seq_len, add_sos_eos=False):
             word_list = self.tokenize(text)
     
             if add_sos_eos:
                 if len(word_list) == seq_len - 2:
                     word_list = [self.sos_token] + word_list + [self.eos_token]
                 elif len(word_list) < seq_len - 2:
                     word_list = [self.sos_token] + word_list + [self.eos_token] + [self.pad_token] * (seq_len - len(word_list) - 2)
                 else:
                     word_list = [self.sos_token] + word_list[:seq_len - 2] + [self.eos_token]
             else:
                 # 补齐或截断到指定的seq_len
                 if len(word_list) > seq_len:
                     word_list = word_list[0:seq_len]
                 elif len(word_list) < seq_len:
                     word_list = word_list + [self.pad_token] * (seq_len - len(word_list))
     
             return [self.word2index.get(word, self.unk_token_id) for word in word_list]
     
         @classmethod
         def from_vocab(cls, vocab_file):
             # 1. 加载词表文件
             with open(vocab_file, 'r', encoding='utf-8') as f:
                 vocab_list = [line[:-1] for line in f.readlines()]
     
             # 2. 创建tokenizer对象
             return cls(vocab_list)
     
         @classmethod
         def build_vocab(cls, sentences, vocab_file):
             # 构建词表（用训练集）
             vocab_set = set()
             for sentence in tqdm(sentences, desc='构建词表'):
                 for word in cls.tokenize(sentence):
                     if word.strip() != '':  # 去除不可见的token
                         vocab_set.add(word)
             vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(vocab_set)
             print(f'词表大小：{len(vocab_list)}')
     
             # 保存词表
             with open(vocab_file, 'w', encoding='utf-8') as f:
                 for word in vocab_list:
                     f.write(word + '\n')
             print('词表保存完成')
     
     
     class ChineseTokenizer(BaseTokenizer):
         @staticmethod
         def tokenize(text):
             return list(text)
     
         def decode(self, word_ids):
             word_list = [self.index2word[word_id] for word_id in word_ids]
             return ''.join(word_list)
     
     
     class EnglishTokenizer(BaseTokenizer):
         @staticmethod
         def tokenize(text):
             return word_tokenize(text)
     
         def decode(self, word_ids):
             word_list = [self.index2word[word_id] for word_id in word_ids]
             return  TreebankWordDetokenizer().detokenize(word_list)
     
     
     if __name__ == '__main__':
         print(ChineseTokenizer.tokenize("我喜欢乘坐地铁。"))
         print(EnglishTokenizer.tokenize("I'm happy."))
         print(EnglishTokenizer.tokenize('I am interested in Japanese history.'))
     ```
   
   - 自定义数据集
   
     ```python
     import pandas as pd
     import torch
     from torch.utils.data import Dataset, DataLoader
     import config
     
     
     # 1. 定义Dataset
     class TranslationDataset(Dataset):
         def __init__(self, data_path):
             self.data = pd.read_json(data_path, orient='records', lines=True).to_dict(orient='records')
     
         def __len__(self):
             return len(self.data)
     
         def __getitem__(self, index):
             input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
             target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
             return input_tensor, target_tensor
     
     
     # 2. 获取DataLoader得方法
     def get_dataloader(train=True):
         data_path = config.PROCESSED_DIR / 'indexed_train.jsonl' if train else config.PROCESSED_DIR / 'indexed_test.jsonl'
         dataset = TranslationDataset(data_path)
         return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
     
     
     if __name__ == '__main__':
         train_dataloader = get_dataloader(train=True)
         print(f'train batch个数：{len(train_dataloader)}')
         test_dataloader = get_dataloader(train=False)
         print(f'test batch个数：{len(test_dataloader)}')
     
         for inputs, targets in train_dataloader:
             print(f'inputs.shape:{inputs.shape}')  # [batch_size, seq_len]
             print(f'targets.shape:{targets.shape}')  # [batch_size,seq_len]
             break
     
     ```
   
   - 模型定义
   
     ```python
     from tensorboard import summary
     from torch import nn
     import torch
     import config
     
     
     class Attention(nn.Module):
         def forward(self, decoder_hidden, encoder_outputs):
             # decoder_hidden.shape: [1,batch_size,decoder_hidden_size]
             # encoder_output.shape: [batch_size, seq_len, 2 * hidden_size]
             attention_score = torch.bmm(decoder_hidden.transpose(0, 1), encoder_outputs.transpose(1, 2))
             # attention_score.shape: [batch_size,1,seq_len]
             attention_weight = torch.softmax(attention_score, dim=-1)
             context_vector = torch.bmm(attention_weight, encoder_outputs)
             # context_vector.shape: [batch_size,1,hidden_size]
             return context_vector
     
     
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
             # output.shape: [batch_size, seq_len, hidden_size * num_directions]
             last_hidden_forward = hidden[-2]
             last_hidden_backward = hidden[-1]
             context_vector = torch.cat([last_hidden_forward, last_hidden_backward], dim=1)
             # context_vector.shape: [batch_size,hidden_size * 2]
             return output, context_vector
     
     
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
                 in_features=2 * config.DECODER_HIDDEN_SIZE,
                 out_features=vocab_size
             )
             self.attention = Attention()
     
         def forward(self, tgt, hidden, encoder_outputs):
             embedded = self.embedding(tgt)  # (batch_size, 1, embedding_dim)
             output, hidden = self.gru(embedded, hidden)  # output: (batch_size, 1, hidden_size)
     
             # 注意力机制
             # encoder_outputs和hidden进行计算，得到一个上下文向量
             context_vector = self.attention(hidden, encoder_outputs)
             # context_vector.shape: [batch_size,1,hidden_size]
     
             # 拼接上下文向量和输出向量
             combined = torch.cat((output, context_vector), dim=2)
             # combined.shape: [batch_size,1,2 * hidden_size]
             output = self.linear(combined)  # (batch_size, 1, vocab_size)
             return output, hidden
     
     ```
   
   - 模型训练
   
     ```python
     import time
     from itertools import chain
     import torch
     from torch.utils.tensorboard import SummaryWriter
     from tqdm import tqdm
     
     from tokenizer import ChineseTokenizer, EnglishTokenizer
     import config
     from model import TranslationDecoder, TranslationEncoder
     from dataset import get_dataloader
     
     
     def train_one_epoch(dataloader, encoder, decoder, optimizer, loss_function, device):
         encoder.train()
         decoder.train()
         epoch_total_loss = 0
         for inputs, targets in tqdm(dataloader, desc='训练'):
             inputs = inputs.to(device)
             # inputs.shape：[batch_size,seq_len]
             targets = targets.to(device)
             # targets.shape: [batch_size,seq_len]
     
             optimizer.zero_grad()
     
             # 编码
             encoder_outputs,context_vector = encoder(inputs)
             # context_vector.shape: [batch_size,encoder_hidden_size]
     
             # 解码
             decoder_input = targets[:, 0:1]
             # decoder_input.shape: [batch_size,1]
             decoder_hidden = context_vector.unsqueeze(0)
             # decoder_hidden.shape: [1,batch_size,decoder_hidden_size]
     
             decoder_outputs = []
             # 1,seq_len次循环
             for t in range(1, targets.shape[1]):
                 decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden,encoder_outputs)
                 # decoder_output.shape: [batch_size, 1, vocab_size]
                 decoder_outputs.append(decoder_output)
                 decoder_input = targets[:,t:t+1]
             # 预测结果
             decoder_outputs = torch.cat(decoder_outputs, dim=1)
             # decoder_outputs.shape: [batch_size, seq_len-1, vocab_size]
             decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
             # decoder_outputs.shape: [batch_size * (seq_len-1), vocab_size]
     
             # 期望值
             decoder_targets = targets[:, 1:]
             # decoder_targets.shape: [batch_size, seq_len-1]
             decoder_targets = decoder_targets.reshape(-1)
             # decoder_targets.shape: [batch_size * (seq_len-1)]
     
             # 计算损失
             loss = loss_function(decoder_outputs, decoder_targets)
     
             loss.backward()
             optimizer.step()
             epoch_total_loss += loss.item()
         return epoch_total_loss / len(dataloader)
     
     
     def train():
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
         # tokenizer
         zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR / 'zh_vocab.txt')
         en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR / 'en_vocab.txt')
     
         # 模型
         encoder = TranslationEncoder(zh_tokenizer.vocab_size, zh_tokenizer.pad_token_id).to(device)
         decoder = TranslationDecoder(en_tokenizer.vocab_size, en_tokenizer.pad_token_id).to(device)
     
         # 加载数据
         dataloader = get_dataloader()
     
         # 损失函数
         loss_function = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)
     
         # 优化器
         optimizer = torch.optim.Adam(params=chain(encoder.parameters(), decoder.parameters()), lr=config.LEARNING_RATE)
     
         # tensorboard
         writer = SummaryWriter(config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))
     
         best_loss = float('inf')
         for epoch in range(1, 1 + config.EPOCHS):
             print(f'======= Epoch {epoch} =======')
             avg_loss = train_one_epoch(dataloader, encoder, decoder, optimizer, loss_function, device)
             print(f'Loss: {avg_loss:.4f}')
             writer.add_scalar('Loss', avg_loss, epoch)
             if avg_loss < best_loss:
                 best_loss = avg_loss
                 torch.save(encoder.state_dict(), config.MODELS_DIR / 'encoder.pt')
                 torch.save(decoder.state_dict(), config.MODELS_DIR / 'decoder.pt')
                 print('模型保存成功')
     
     
     if __name__ == '__main__':
         train()
     
     ```
   
   - 预测模型
   
     ```python
     import torch
     
     from tokenizer import ChineseTokenizer, EnglishTokenizer
     import config
     from model import TranslationEncoder, TranslationDecoder
     
     
     def predict_batch(input_tensor, encoder, decoder, ch_tokenizer, en_tokenizer, device):
         """
         批量预测
         :param input_tensor: 一批中文句子 [batch_size, seq_len]
         :param encoder:
         :param decoder:
         :param ch_tokenizer:
         :param en_tokenizer:
         :param device:
         :return: 一批与之对应的英文句子 [[],[],...]
         """
         encoder.eval()
         decoder.eval()
         with torch.no_grad():
             # 编码
             encoder_outputs,context_vector = encoder(input_tensor)
             # context_vector.shape: [batch_size,decoder_hidden_size]
             # 解码
             batch_size = input_tensor.shape[0]
             decoder_input = torch.full((batch_size, 1), en_tokenizer.sos_token_id, device=device)
             # decoder_input.shape: [batch_size,1]
             decoder_hidden = context_vector.unsqueeze(0)
             # decoder_hidden: [1,batch_size,decoder_hidden_size]
     
             generated = [[] for _ in range(batch_size)]
             is_finished = [False for _ in range(batch_size)]
             for t in range(config.SEQ_LEN):
                 decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden,encoder_outputs)
                 # decoder_output.shape: [batch_size,1,vocab_size]
                 predict_indexes = torch.argmax(decoder_output, dim=-1, keepdim=False)
                 # 处理每个时间步的预测结果
                 for i in range(batch_size):
                     if is_finished[i]:
                         continue
                     else:
                         if predict_indexes[i].item() == en_tokenizer.eos_token_id:
                             is_finished[i] = True
                         else:
                             generated[i].append(predict_indexes[i].item())
                 if all(is_finished):
                     break
                 decoder_input = predict_indexes
             return generated
     
     
     def predict(user_input, encoder, decoder, ch_tokenizer, en_tokenizer, device):
         # 处理数据
         index_list = ch_tokenizer.encode(user_input, config.SEQ_LEN)
         input_tensor = torch.tensor([index_list]).to(device)
         # input_tensor.shape: (batch_size,seq_len)
         batch_result = predict_batch(input_tensor, encoder, decoder, ch_tokenizer, en_tokenizer, device)
         result = batch_result[0]
         return en_tokenizer.decode(result)
     
     
     def run_predict():
         # 准备资源
         # 设备
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         # tokenizer
         ch_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR / 'zh_vocab.txt')
         en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR / 'en_vocab.txt')
         # 模型
         encoder = TranslationEncoder(vocab_size=ch_tokenizer.vocab_size, padding_index=ch_tokenizer.pad_token_id).to(device)
         encoder.load_state_dict(torch.load(config.MODELS_DIR / 'encoder.pt'))
         decoder = TranslationDecoder(vocab_size=en_tokenizer.vocab_size, padding_index=en_tokenizer.pad_token_id).to(device)
         decoder.load_state_dict(torch.load(config.MODELS_DIR / 'decoder.pt'))
         # 运行测试
         while True:
             user_input = input("中文：")
             if user_input in ['q', 'quit']:
                 break
             if user_input.strip() == '':
                 continue
             result = predict(user_input, encoder, decoder, ch_tokenizer, en_tokenizer, device)
             print('英文：' + result)
     
     
     if __name__ == '__main__':
         run_predict()
     
     ```
   
   - 评估模型
   
     ```python
     import torch
     from nltk.translate.bleu_score import corpus_bleu
     from tqdm import tqdm
     
     from tokenizer import ChineseTokenizer, EnglishTokenizer
     import config
     from model import TranslationEncoder, TranslationDecoder
     from dataset import get_dataloader
     from predict import predict_batch
     
     
     def evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device):
         references = []  # [[[4,5,6,7]],[[5,6,7,8,9]],[[7,8,9]]]
         predictions = []  # [[4,5,6,7],[5,6,7,8,9],[7,8,9]]
     
         special_tokens = [en_tokenizer.sos_token_id, en_tokenizer.eos_token_id, en_tokenizer.pad_token_id]
     
         for inputs, targets in tqdm(dataloader, desc='评估'):
             inputs = inputs.to(device)
             # inputs.shape: [batch_size, seq_len]
     
             targets = targets.tolist()
             # 参考译文: [[*,*,*,*,*,*],[*,*,*,*,*,*],[*,*,*,*,*,*]]
     
             batch_result = predict_batch(inputs, encoder, decoder, zh_tokenizer, en_tokenizer, device)
             # 预测译文：batch_result.shape: [[4,6,7],[11,23,45,78,99],[88,99,26,55]...]
     
             # 处理预测结果
             predictions.extend(batch_result)
     
             # 获取参考译文
             references.extend([[[index for index in target if index not in special_tokens]] for target in targets])
     
         return corpus_bleu(references, predictions)
     
     
     def run_evaluate():
         # 准备资源
         # 设备
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
         # tokenizer
         zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR / 'zh_vocab.txt')
         en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR / 'en_vocab.txt')
     
         # 模型
         encoder = TranslationEncoder(vocab_size=zh_tokenizer.vocab_size,
                                      padding_index=zh_tokenizer.pad_token_id).to(device)
         encoder.load_state_dict(torch.load(config.MODELS_DIR / 'encoder.pt'))
     
         decoder = TranslationDecoder(vocab_size=en_tokenizer.vocab_size,
                                      padding_index=en_tokenizer.pad_token_id).to(device)
         decoder.load_state_dict(torch.load(config.MODELS_DIR / 'decoder.pt'))
     
         # 加载数据集
         dataloader = get_dataloader(train=False)
     
         bleu = evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device)
     
         print(f'Bleu: {bleu}')
     
     
     if __name__ == '__main__':
         run_evaluate()
     
     ```
   
   - 配置文件
   
     ```python
     from pathlib import Path
     
     ROOT_DIR = Path(__file__).parent.parent
     
     RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
     PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
     LOGS_DIR = ROOT_DIR / 'logs'
     MODELS_DIR = ROOT_DIR / 'models'
     
     SEQ_LEN = 32
     BATCH_SIZE = 128
     EMBEDDING_DIM = 128
     ENCODER_HIDDEN_SIZE = 256
     DECODER_HIDDEN_SIZE = ENCODER_HIDDEN_SIZE * 2
     ENCODER_LAYERS = 1
     LEARNING_RATE = 1e-3
     EPOCHS = 20
     
     ```

### 5.5 存在问题

尽管注意力机制极大地增强了 Seq2Seq 模型的建模能力，但由于其核心依然依赖于 RNN 结构，仍面临两个根本性问题：

- 计算过程无法并行

  RNN 的时间步之间存在强依赖，必须顺序执行，限制了训练效率和硬件资源的利用率。

- 长期依赖问题仍未根除

  模型需要跨多个时间步传递信息，对于超长序列，训练过程中容易出现梯度消失，难以有效建模长距离依赖关系。