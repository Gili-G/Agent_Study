import math
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, dim, num_heads, N):
        super(Encoder,self).__init__()

        self.position_embedding = PositionEmdedding()
        self.multi_head_attention = MultiHeadAttention(num_heads, dim)
        self.feed_forward = FeedForward()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):
        x = self.position_embedding(x)
        for _ in range(self.N):
            x = self.norm(self.dropout(self.multi_head_attention(x)) + x)
            x = self.norm(self.dropout(self.feed_forward(x)) + x)

        return x

class FeedForward(nn.Moudle):
    def __int__(self, dim):
        super(FeedForward,self).__int__()

        self.fc1 = nn.Linear(dim, dim*8)
        self.fc2 = nn.Linear(dim*8, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x -> fc1 -> relu -> dropout -> fc2
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class MultiHeadAttention(nn.Module):
    def __int__(self, num_heads, dim):
        super(MultiHeadAttention,self).__int__()
        assert dim//num_heads == 0
        self.dim_k = dim/num_heads

        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.Out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q_heads = Q.view(B, N, -1, C/self.num_heads).transpose(1, 2)
        K_heads = Q.view(B, N, -1, C/self.num_heads).transpose(1, 2)
        V_heads = Q.view(B, N, -1, C/self.num_heads).transpose(1, 2)

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

