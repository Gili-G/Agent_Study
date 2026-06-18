import math
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len, dim, num_heads, N, dropout=0.1):
        super(Encoder,self).__init__()

        self.position_embedding = PositionEmbedding(dim, seq_len)
        self.multi_head_attention = MultiHeadAttention(num_heads, dim)
        self.feed_forward = FeedForward(dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.N = N

    def forward(self, x):
        x = self.position_embedding(x)
        for _ in range(self.N):
            x = self.norm(self.dropout(self.multi_head_attention(x,x,x)) + x)
            x = self.norm(self.dropout(self.feed_forward(x)) + x)

        return x

class Decoder(nn.Module):
    def __init__(self, seq_len, dim, num_heads, N, dropout=0.1):
        super(Decoder, self).__init__()

        self.position_embedding = PositionEmbedding(dim, seq_len)
        self.multi_head_attention = MultiHeadAttention(num_heads, dim)
        self.feed_forward = FeedForward(dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.N = N

    def forward(self, x, encoder_out):
        x = self.position_embedding(x)
        for _ in range(self.N):
            x = self.norm(self.dropout(self.multi_head_attention(x,x,x,mask=True)) + x)
            x = self.norm(self.dropout(self.multi_head_attention(x,encoder_out,encoder_out)) + x)
            x = self.norm(self.dropout(self.feed_forward(x)) + x)

        return x
    
class PositionEmbedding(nn.Module):
    def __init__(self, dim, seq_len, dropout=0.1):
        super(PositionEmbedding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(seq_len).unsqueeze(1)
        term = torch.exp(torch.arange(0,dim,2)*(-torch.log(10000)/dim))

        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(pos * term)
        pe[:, 1::2] = torch.cos(pos * term)

        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        x = self.dropout(x)

        return x


class FeedForward(nn.Moudle):
    def __init__(self, dim, dropout=0.1):
        super(FeedForward,self).__int__()

        self.fc1 = nn.Linear(dim, dim*8)
        self.fc2 = nn.Linear(dim*8, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x -> fc1 -> relu -> dropout -> fc2
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim):
        super(MultiHeadAttention,self).__int__()
        assert dim//num_heads == 0
        self.dim_k = dim/num_heads

        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.Out = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        B, N, C = q.shape

        Q = self.W_Q(q)
        K = self.W_K(k)
        V = self.W_V(v)

        Q_heads = Q.view(B, N, -1, C/self.num_heads).transpose(1, 2)
        K_heads = K.view(B, N, -1, C/self.num_heads).transpose(1, 2)
        V_heads = V.view(B, N, -1, C/self.num_heads).transpose(1, 2)

        if mask is not None:
            attn_out, attn_weights = self.scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask=True)
        else:
            attn_out, attn_weights = self.scaled_dot_product_attention(Q_heads, K_heads, V_heads)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        attn_out = self.Out(attn_out)

        return attn_out, attn_weights

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_weights = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(self.dim_k)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(attn_weights,dim=-1)
        attn_out = torch.matmul(attn_weights,v)

        return attn_out, attn_weights

