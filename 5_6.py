# [프로그램 5-11] 토이 번역 데이터 구성하기
import torch
from torch.utils.data import Dataset, DataLoader

pairs = [
    ("i love you", "나는 너를 사랑해"),
    ("he is happy", "그는 행복하다"),
    ("she is a student", "그녀는 학생이다"),
    ("thank you", "고마워"),
    ("good morning", "좋은 아침")
]

# 간단한 토크나이저
def tokenize(sentence):
    return sentence.split()

# 단어 사전 구성
src_vocab = {"<pad>":0, "<sos>":1, "<eos>":2}
tgt_vocab = {"<pad>":0, "<sos>":1, "<eos>":2}

for src, tgt in pairs:
    for w in tokenize(src):
        if w not in src_vocab:
            src_vocab[w] = len(src_vocab)
    for w in tokenize(tgt):
        if w not in tgt_vocab:
            tgt_vocab[w] = len(tgt_vocab)

inv_tgt = {v:k for k,v in tgt_vocab.items()}

max_len = 8

def encode(sentence, vocab):
    tokens = ["<sos>"] + tokenize(sentence) + ["<eos>"]
    tokens = tokens[:max_len] + ["<pad>"]*(max_len - len(tokens))
    return torch.tensor([vocab[t] for t in tokens])

class ToyTranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_encoded = encode(src, src_vocab)     # shape: (max_len,)
        tgt_encoded = encode(tgt, tgt_vocab)     # shape: (max_len,)
        return src_encoded, tgt_encoded

dataset = ToyTranslationDataset(pairs)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# [프로그램 5-12] Bahdanau 어텐션을 파이토치 모듈로 구현하기
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V  = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (B, H)
        # encoder_outputs: (B, T, H)
        # score: (B, T, 1)
        score = self.V(torch.tanh(
            self.W1(encoder_outputs) + self.W2(decoder_hidden).unsqueeze(1)
        ))
        attn_weights = F.softmax(score, dim=1)         # (B, T, 1)
        context = (attn_weights * encoder_outputs).sum(dim=1)   # (B, H)
        return context, attn_weights

# [프로그램 5-13] 인코더-디코더 모델 구성하기 
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm  = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x: (B, T)
        x = self.embed(x)                 # (B, T, E)
        outputs, (h, c) = self.lstm(x)    # outputs: (B, T, H)
        return outputs, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, attention):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm  = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.attn  = attention
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (B,) 단일 토큰
        x = self.embed(x).unsqueeze(1)      # (B,1,E)
        
        # 디코더 현재 은닉 상태
        decoder_hidden = hidden[-1]         # (B,H)
        
        # 어텐션 계산
        context, _ = self.attn(decoder_hidden, encoder_outputs)  # (B,H)
        context = context.unsqueeze(1)       # (B,1,H)
        
        # LSTM 입력 결합
        lstm_input = torch.cat([x, context], dim=2)  # (B,1,E+H)
        
        outputs, (h, c) = self.lstm(lstm_input, (hidden, cell))  # outputs: (B,1,H)
        logits = self.fc(outputs.squeeze(1))                     # (B,V)
        return logits, (h, c)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoder_outputs, (h, c) = self.encoder(src)
        
        batch_size = src.size(0)
        T = tgt.size(1)
        
        outputs = []
        token = tgt[:, 0]     # <sos>
        
        for t in range(1, T):
            logits, (h, c) = self.decoder(token, h, c, encoder_outputs)
            outputs.append(logits.unsqueeze(1))
            token = tgt[:, t]     # teacher forcing
        
        return torch.cat(outputs, dim=1)   # (B, T-1, V)

# [프로그램 5-14] 학습 루프 실행을 위한 학습 준비하기
embed_size = 32
hidden_size = 64

attention = BahdanauAttention(hidden_size)
encoder    = Encoder(len(src_vocab), embed_size, hidden_size)
decoder    = Decoder(len(tgt_vocab), embed_size, hidden_size, attention)
model      = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# [프로그램 5-15] 학습 루프 실행하기
num_epochs = 30

for epoch in range(num_epochs):
    total_loss = 0.0
    for src, tgt in loader:
        optimizer.zero_grad()
        logits = model(src, tgt)                # (B, T-1, V)
        
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1)              # <sos> 다음 토큰부터 비교
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# [프로그램 5-16] 문장 번역하기
def translate(sentence):
    src = encode(sentence, src_vocab).unsqueeze(0)
    encoder_outputs, (h, c) = encoder(src)
    
    token = torch.tensor([tgt_vocab["<sos>"]])
    result = []
    
    for _ in range(max_len):
        logits, (h, c) = decoder(token, h, c, encoder_outputs)
        token = logits.argmax(dim=1)
        word = inv_tgt[token.item()]
        if word == "<eos>":
            break
        result.append(word)
    return " ".join(result)

print(translate("i love you"))
print(translate("good morning"))

