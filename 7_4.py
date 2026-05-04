# [프로그램 7-1] 스케일드 닷-프로덕트 어텐션 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V shape: (batch, heads, seq_len, head_dim)
    # 점수 행렬: (batch, heads, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

    if mask is not None:
        # mask가 True인 위치를 -inf로 채워 소프트맥스에서 확률이 0에 가깝게 되도록 처리한다.
        scores = scores.masked_fill(mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)                      # (batch, heads, seq_len_q, seq_len_k)
    out = torch.matmul(attn, V)                           # (batch, heads, seq_len_q, head_dim)
    return out, attn

# [프로그램 7-2] 멀티-헤드 어텐션 구현하기
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        assert embed_dim % heads == 0
        self.head_dim = embed_dim // heads
        self.heads = heads

        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)
        self.WO = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, kv=None):
        """
        x: 쿼리를 구성하는 입력, shape (batch, seq_len_q, embed_dim)
        kv: 키·값을 구성하는 입력, cross-attention 시 인코더 출력이 들어온다.
            None이면 셀프 어텐션으로 동작한다.
        """
        K_input = x if kv is None else kv

        batch, seq_len_q, embed_dim = x.shape
        seq_len_k = K_input.size(1)

        # 선형 변환 후 (batch, heads, seq_len, head_dim) 형태로 변환
        Q = self.WQ(x).view(batch, seq_len_q, self.heads, self.head_dim).transpose(1, 2)
        K = self.WK(K_input).view(batch, seq_len_k, self.heads, self.head_dim).transpose(1, 2)
        V = self.WV(K_input).view(batch, seq_len_k, self.heads, self.head_dim).transpose(1, 2)

        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        # 다시 (batch, seq_len_q, embed_dim)으로 되돌린다.
        out = out.transpose(1, 2).contiguous().view(batch, seq_len_q, embed_dim)
        return self.WO(out)

# [프로그램 7-3] 인코더 레이어 구현하기
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)

        # 셀프 어텐션 + 잔차 연결 + 정규화
        h = self.attn(x)
        x = self.norm1(x + h)

        # 피드포워드 + 잔차 연결 + 정규화
        h2 = self.ff(x)
        x = self.norm2(x + h2)
        return x

# [프로그램 7-4] 디코더 레이어 구현하기
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = MultiHeadAttention(embed_dim, heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_out, mask=None):
        # x: 디코더 입력(부분 생성 시퀀스), enc_out: 인코더 출력

        # 마스크드 셀프 어텐션
        h = self.self_attn(x, mask=mask)
        x = self.norm1(x + h)

        # 인코더–디코더 어텐션
        h2 = self.cross_attn(x, kv=enc_out)
        x = self.norm2(x + h2)

        # 피드포워드
        h3 = self.ff(x)
        x = self.norm3(x + h3)
        return x

# [프로그램 7-5] 트랜스포머 모델 구성하기
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, heads, ff_dim, depth, max_len=128):
        super().__init__()
        # 토큰 임베딩
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        # 위치 임베딩 (학습 가능한 벡터)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        # 인코더와 디코더 레이어 스택
        self.enc_layers = nn.ModuleList([
            EncoderLayer(embed_dim, heads, ff_dim) for _ in range(depth)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(embed_dim, heads, ff_dim) for _ in range(depth)
        ])

        # 최종 출력층: 임베딩을 어휘 분포로 변환
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt, tgt_mask=None):
        # src, tgt shape: (batch, seq_len)
        batch, src_len = src.shape
        _,    tgt_len = tgt.shape

        device = src.device

        # 위치 인덱스 생성: 0, 1, ..., seq_len-1
        src_positions = torch.arange(src_len, device=device).unsqueeze(0)  # (1, src_len)
        tgt_positions = torch.arange(tgt_len, device=device).unsqueeze(0)  # (1, tgt_len)

        # 토큰 임베딩 + 위치 임베딩 결합
        src_emb = self.tok_embed(src) + self.pos_embed(src_positions)      # (batch, src_len, embed_dim)
        tgt_emb = self.tok_embed(tgt) + self.pos_embed(tgt_positions)      # (batch, tgt_len, embed_dim)

        # 인코더
        enc = src_emb
        for layer in self.enc_layers:
            enc = layer(enc)

        # 디코더
        dec = tgt_emb
        for layer in self.dec_layers:
            dec = layer(dec, enc, mask=tgt_mask)

        # 디코더 출력 → 각 위치에서의 어휘 분포
        return self.fc_out(dec)  # (batch, tgt_len, vocab_size)

# [프로그램 7-6] 마스크 생성하고 트랜스포머 모델 실행하기
def generate_square_subsequent_mask(seq_len):
    """
    디코더용 마스크 생성 함수.
    현재 시점 이후의 위치를 True로 표시하여 어텐션 계산에서 제외한다.
    반환 shape: (1, 1, seq_len, seq_len)  → (batch, heads, seq_q, seq_k)에 브로드캐스트 가능
    """
    # 상삼각(대각선 위쪽)을 1로 채운 뒤 bool로 변환
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # 배치 차원, 헤드 차원을 위한 축 추가
    return mask.unsqueeze(0).unsqueeze(0)


# 간단한 동작 테스트
batch = 2
src = torch.randint(0, 50, (batch, 6))   # (batch, src_len)
tgt = torch.randint(0, 50, (batch, 6))   # (batch, tgt_len)
tgt_mask = generate_square_subsequent_mask(tgt.size(1))

model = Transformer(
    vocab_size=50,
    embed_dim=32,
    heads=4,
    ff_dim=64,
    depth=2,
    max_len=64
)

out = model(src, tgt, tgt_mask=tgt_mask)
print(out.shape)   # 예상 출력: (batch, tgt_len, vocab_size)

