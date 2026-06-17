import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, dim, num_heads, N):
        super(Encoder,self).__init__()
        self.input_dim = input_dim
        self.dim = dim 
        self.num_heads = num_heads
        self.N = N

        self.position_embedding = PositionEmdedding()
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.norm = nn.layernorm(dim)

    def forward(self, x):
        x = self.position_embedding(x)
        for _ in range(self.N):
            x = self.norm(self.multi_head_attention(x) + x)
            x = self.norm(self.feed_forward(x) + x)


class MultiHeadAttention(nn.Module):
    def __int__(self, num_heads, dim):
        super(MultiHeadAttention,self).__int__()
        assert dim//num_heads == 0

        self.dim = dim 
        self.num_heads = num_heads
        self.dim_k = dim/num_heads

        self.W_Q = nn.Linear(self.dim, self.dim)
        self.W_K = nn.Linear(self.dim, self.dim)
        self.W_V = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        B, N, C = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q_heads = Q.view(B, N, -1, C/self.num_heads).transpose(1, 2)
        K_heads = Q.view(B, N, -1, C/self.num_heads).transpose(1, 2)
        V_heads = Q.view(B, N, -1, C/self.num_heads).transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn = F.softmax(torch.matmul(q,k.transpose(2,3))/math.sqrt(self.dim_k),dim=-1)

