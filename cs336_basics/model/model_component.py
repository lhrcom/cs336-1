import torch
import torch.nn as nn
from einops import einsum,rearrange

class Linear(nn.Module):
    def __init__(self,in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... in_features, out_features in_features -> ... out_features')

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g= nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # (batch_size, sequence_length, d_model)
        RMS = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / RMS * self.g
        return result.to(in_dtype)

class positionwise_feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        return self.linear2((self.linear1(x) * torch.sigmoid(self.linear1(x))) * self.linear3(x))

        
# RoPE
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__() 
        assert d_k % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        # 位置 t × 第 k 个频率：
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        out = torch.stack((rot_even, rot_odd), dim=-1)
        return rearrange(out,'... seq d_k d -> ... seq (d_k d)')


def softmax(x):
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    scores = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys") * (k.shape[-1] ** -0.5)
    # scores = einsum(q, k, "... seq d_k, ... seq d_k -> ... seq") * (k.shape[-1] ** -0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    att = softmax(scores) @ v
    return att

class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None):
        super().__init__()
        d_k = d_model // num_heads
        self.num_heads = num_heads
        self.query = Linear(d_model,d_k*num_heads)
        self.key = Linear(d_model,d_k*num_heads)
        self.value = Linear(d_model,d_k*num_heads)
        self.o = Linear(d_k*num_heads,d_model)
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
        else:
            self.rope = None
    def forward(self,x,token_positions=None):
        T = x.shape[-2]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = rearrange(q,'... seq (num_heads d_k) -> ... num_heads seq d_k ', num_heads = self.num_heads)
        k = rearrange(k,'... seq (num_heads d_k) -> ... num_heads seq d_k ', num_heads = self.num_heads)
        v = rearrange(v,'... seq (num_heads d_k) -> ... num_heads seq d_k ', num_heads = self.num_heads)

        if self.rope is not None:
            seq_len = x.shape[-2]
            
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # 在 seq 前面插一个长度为 1 的维度，用来对齐 Q 的 head 维
            pos = token_positions.unsqueeze(-2)        # (..., seq) -> (..., 1, seq)
            q = self.rope(q, pos)                      # Q: (..., h, seq, d_k)
            k = self.rope(k, pos)

        casual_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = scaled_dot_product_attention(q, k, v, casual_mask)
        out = self.o(rearrange(att, '... num_heads seq d_k -> ... seq (num_heads d_k)'))
        return out
                

class transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len,device=None,dtype=None):
        super().__init__()
        self.attn = multihead_self_attention(d_model, num_heads, theta, max_seq_len)
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ffn = positionwise_feedforward(d_model,d_ff,device=device,dtype=dtype)
    def forward(self,x,token_positions=None):
        y = x + self.attn(self.ln1(x), token_positions)
        out = y + self.ffn(self.ln2(y))
        return out

class transformer_lm(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            transformer_block(d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        h = self.token_embeddings(x)
        # x.shape:(b,seq,d_model)
        for layer in self.layers:
            h = layer(h, token_positions)
        # x.shape:(b,seq,d_model)
        return self.lm_head(self.ln_final(h))
        # x.shape:(b,seq,vocab_size)



# Consider a GPT-2 XL-sized model using our assignment architecture, which has the following
# configuration:
# vocab_size: 50,257
# context_length: 1,024
# num_layers: 48
# d_model: 1,600
# num_heads: 25
# d_ff: 4,288 (the nearest multiple of 64 to 8/3 × 1600)
# Suppose we constructed our model using this configuration. How many trainable parameters
# would our model have? Assuming each parameter is represented using single-precision floating
# point, how much memory is required to just load this model?

'''
trainable parameters:
1. Embedding: vocab_size * d_model = 50,257 * 1,600 = 80411200
2. transformer_block:
    attention KQVO = d_model * d_model * 4 = 1,600 * 1,600 * 4 = 10240000
    RMSNorm: d_model * 2 = 1,600 * 2 = 3200
    FFN: d_model -> d_ff -> d_model = d_model * d_ff * 3 = 1,600 * 4,288 * 3 = 20582400
    so total trainable parameters in one block  = 4 * d_model * d_model + d_model * 2 + d_model * d_ff * 3 = 30825600
    so total : 30825600 * num_layers = 30825600 * 48 = 1479628800
3. RMSNorm: 1600
4. lm_head = d_model * vocab_size = 1,600 * 50,257 = 80411200

The total trainable parameters is 1.64B
Single-precision floating-point numbers (fp32) take up 4 bytes (32 bits) each. -> 4 * 1.64B = 6.56 GB (memory)
'''

'''
How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

if I have a[m,n] b[n,p], The FLOPs of a @ b is 2mnp


A list of matrix multiplies:
transformer_block:
KQVO : (context_length,d_model)(d_model,d_model) * 2 * 4 = 20971520000 
QKᵀ : (context_length,d_model)(d_model,context_length) * 2 = 3355443200
attn_weight * V = (context_length,context_length)(context_length,d_model) * 2 = 3355443200
FFN Linear1 (context_length,d_model)(d_model,d_ff) * 2 = 14050918400
FFN Linear2 (context_length,d_ff)(d_ff,d_model) * 2 = 14050918400
FFN linear3 (context_length,d_model)(d_model,d_ff) * 2 = 14050918400
transformer_block * num_layers = 3352087756800 FLOPs
lm_head = (context_length,d_model)(d_model, vocab_size) * 2 = 1646451097600 FLOPs
total FLOPs = 4998538854400 FLOPs = 4.99 TFLOPs
'''

'''
Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium
(24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As
the model size increases, which parts of the Transformer LM take up proportionally more or
less of the total FLOPs?

	               small	medium	large	XL
QKVO 投影	        16.6%	20.0%	 21.4%	22.3%
QKᵀ + 注意力权重@V	 11.1%	 10.0%	  8.6%	 7.1%
FFN	               49.8%    59.9%	64.2%  66.9%
lm_head	           22.6%   10.2%	5.8%   3.6%
合计 (TFLOPs)	    0.35	1.03	 2.26	4.51
'''

'''
Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one
forward pass change? How does the relative contribution of FLOPs of the model components
change?

部分	                公式	    T=1024	               占比	    T=16384	            占比	    增长倍数
QKVO 投影	            8LTd²	   1,006,632,960,000	28.6%	16,106,127,360,000	   12.1%	    16×
QKᵀ + 注意力权重×V	     4LT²d	    322,122,547,200	      9.2%	  82,463,372,083,200	61.7%	    256×
FFN	                   6LT·d·d_ff  2,023,332,249,600	57.5%	32,373,315,993,600	  24.2%	        16×
lm_head	                2TdV	   164,682,137,600	    4.7%	2,634,914,201,600	   2.0%	        16×
合计		                      3,516,769,894,400	            33,577,729,638,400		            约 38×

T(context_length)的增大会出现计算和显存瓶颈(4LT²d), 这也是FlashAttention, 线性注意力等方法要解决的问题
'''