import torch
import math
import torch.nn.functional as F
import torch.nn as nn

def scaled_dot_product_attention(q, k, v, mask=None):
    numerator = q @ torch.transpose(k, -2, -1) # May have to fix this transpose
    if mask is not None:
        numerator = numerator.permute(1, 0, 2, 3) + mask
        numerator = numerator.permute(1, 0, 2, 3)
    denominator = math.sqrt(k.shape[-1])
    attn = F.softmax((numerator/denominator), dim=-1, dtype=torch.float32)
    result = attn @ v
    return result, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.head_dim = d_model // heads # Embed dim must be divisible by heads
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.linear_out = nn.Linear(self.d_model, self.d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_length, _ = q.size()
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q, k, v = [x.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1,2) for x in [q,k,v]]
        values, attn = scaled_dot_product_attention(q, k, v, mask)
        x = values.transpose(1,2).reshape(batch_size, seq_length, self.heads * self.head_dim)
        x = self.linear_out(x)
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, drop_prob=0.1): # Max seq length is set to 50
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=drop_prob)
        self.max_seq_len = max_seq_len
        
        # Calculate denominator, it's the same for even and odd dimensions so you can reuse it
        evens = torch.arange(0, self.d_model, 2).float()
        denom = torch.pow(10000, evens/self.d_model)
        
        # Calculate positional encodings
        self.pe = torch.zeros(self.max_seq_len, self.d_model)
        positions = torch.arange(0, self.max_seq_len).float().reshape(self.max_seq_len, 1)
        
        self.pe[:, 0::2] = torch.sin(positions / denom)
        self.pe[:, 1::2] = torch.cos(positions / denom)
        self.pe = self.pe.unsqueeze(0)
        
    def forward(self, x):
        x = x + self.pe
        x = self.dropout(x)
        
        return x


class LayerNormalization(nn.Module):
    def __init__(self, parameter_shape, eps=1e-5):
        super().__init__()
        self.parameter_shape = parameter_shape
        self.eps = eps
        
        # Define layer norm learnable parameters
        self.gamma = nn.Parameter(torch.ones(parameter_shape))
        self.beta = nn.Parameter(torch.zeros(parameter_shape))
        
    def forward(self, inputs):
        # The layer norm is computed based on each matrix of the batch, not across the batch.
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        
        norm = (self.gamma * ((inputs - mean) / (std + self.eps))) + self.beta
        
        return norm


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.hidden = hidden
        self.drop_prob = drop_prob
        
        self.attn = MultiHeadAttention(self.heads, self.d_model)
        self.norm1 = LayerNormalization(self.d_model)
        self.drop1 = nn.Dropout(p=drop_prob)
        self.pwff = PositionWiseFeedForward(self.d_model, self.hidden, self.drop_prob)
        self.norm2 = LayerNormalization(self.d_model) # Might have to change this
        self.drop2 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, mask):
        residual_x = x.clone()
        x = self.attn(x, x, x, mask=mask)
        x = self.norm1(residual_x + x)
        x = self.drop1(x)
        residual_x = x.clone()
        x = self.pwff(x)
        x = self.norm2(residual_x + x)
        x = self.drop2(x)
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, heads, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.hidden = hidden
        self.drop_prob = drop_prob
        
        self.mask_attn = MultiHeadAttention(self.heads, self.d_model)
        self.norm1 = LayerNormalization(self.d_model)
        self.drop1 = nn.Dropout(p=drop_prob)
        self.cross_attn = MultiHeadAttention(self.heads, self.d_model)
        self.norm2 = LayerNormalization(self.d_model)
        self.drop2 = nn.Dropout(p=drop_prob)
        self.pwff = PositionWiseFeedForward(self.d_model, self.hidden, self.drop_prob)
        self.norm3 = LayerNormalization(self.d_model) # Might have to change this
        self.drop3 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, y, self_mask, cross_mask):
        residual_x = x.clone()
        x = self.mask_attn(x, x, x, mask=self_mask)
        x = self.norm1(residual_x + x)
        x = self.drop1(x)
        residual_x = x.clone()
        x = self.cross_attn(x, y, y, mask=cross_mask) # FINISH THIS 
        x = self.norm2(residual_x + x)
        x = self.drop2(x)
        residual_x = x.clone()
        x = self.pwff(x)
        x = self.norm2(residual_x + x)
        x = self.drop2(x)
        
        return x


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, mask = inputs
        for module in self._modules.values():
            out = module(x, mask)
        return out


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_mask, cross_mask = inputs
        for module in self._modules.values():
            out = module(x, y, self_mask, cross_mask)
        return out


class Encoder(nn.Module):
    def __init__(self, heads, d_model, hidden, num_layers):
        super().__init__()
        self.layers = SequentialEncoder(*[EncoderLayer(heads, d_model, hidden) for _ in range(num_layers)])
        
    def forward(self, x, mask):
        x = self.layers(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, heads, d_model, hidden, num_layers):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(heads, d_model, hidden) for _ in range(num_layers)])
        
    def forward(self, x, y, self_mask, cross_mask):
        x = self.layers(x, y, self_mask, cross_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, max_sequence_length, src_vocab_size, tgt_vocab_size,
                 num_layers, heads, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.src_embed = Embeddings(src_vocab_size, d_model)
        self.tgt_embed = Embeddings(tgt_vocab_size, d_model)
        
        self.enc_pe = PositionalEncoding(d_model, max_sequence_length, drop_prob)
        self.dec_pe = PositionalEncoding(d_model, max_sequence_length, drop_prob)
        
        self.encoder = Encoder(heads, d_model, hidden, num_layers)
        self.decoder = Decoder(heads, d_model, hidden, num_layers)
        
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
    
    def forward(self, src, tgt, enc_self_mask, dec_self_mask, dec_cross_mask):
        x = self.src_embed(src)
        y = self.tgt_embed(tgt)
        
        x = self.enc_pe(x)
        y = self.dec_pe(y)
        
        enc = self.encoder(x, enc_self_mask)
        dec = self.decoder(y, enc, dec_self_mask, dec_cross_mask)
        
        out = self.linear(dec)
        
        return out