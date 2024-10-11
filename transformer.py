# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from torch import Tensor

a = torch.Tensor([[9,1],[3,7]])
print(torch.softmax(a, dim =  -1))

''' Tensor shape: (batch_size, sequence_length) '''
vocab_size = 5755
#vocab_size = 5530

def generate_src_mask(src):
    src_mask = (src != 0).unsqueeze(1).to(src.device)
    return src_mask

def generate_tgt_mask(tgt):
    tgt_mask = (tgt != 0).unsqueeze(1)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
    tgt_mask = tgt_mask & nopeak_mask
    return tgt_mask.to(tgt.device)



class AttentionHead(nn.Module):
    def __init__(self, dim_in, d_model):
        super().__init__()
        self.q = nn.Linear(dim_in, d_model)
        self.k = nn.Linear(dim_in, d_model)
        self.v = nn.Linear(dim_in, d_model)
        self.atten_map = None

    def forward(self, query, key, value, mask = None):
        return self.scaled_dot(self.q(query), self.k(key), self.v(value), mask)
    
    def scaled_dot(self, query, key, value, mask = None):
        temp = query.bmm(key.transpose(1, 2))
        scale = query.size(-1) ** 0.5
        scores = temp / scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        qk_scores = f.softmax(scores, dim=-1)
        self.atten_map = qk_scores
        return qk_scores.bmm(value)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_in, d_model):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, d_model) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * d_model, dim_in)

    def forward(self, query, key, value, mask = None):
        return self.linear(
            torch.cat([h(query, key, value, mask) for h in self.heads], dim=-1)
        )
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1,max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self, d_in,d_model, num_heads, d_ff, dropout, relativeP = False):
        super(EncoderLayer, self).__init__()
        self.self_attn = RelativeMultiHeadAttention(d_model=d_model,n_heads=num_heads) \
            if relativeP else MultiHeadAttention(num_heads=num_heads,dim_in=d_in,d_model=d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, mask = None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output)) #([32, 10, 2048])  [320, 2048])
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        max_l: int,
        dim_in: int = 512,
        dim_model: int = 512,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        rela_P: bool = False
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim_in)
        self.positional_encoding = PositionalEncoding(d_model=dim_in,max_len=max_l)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(dim_in, dim_model, num_heads, dim_feedforward, dropout,relativeP=rela_P)
                for _ in range(num_layers)
            ]
        )
        self.use_relative_p = rela_P

    def forward(self, src: Tensor) -> Tensor:
        m1 = generate_src_mask(src)
        if self.use_relative_p:
            p_encode = self.embedding(src)
        else:
            p_encode = self.positional_encoding(self.embedding(src))
        for layer in self.layers:
            p_encode = layer(p_encode,m1)

        return p_encode
    
class DecoderLayer(nn.Module):
    def __init__(self, d_in,d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads=num_heads,dim_in=d_in,d_model=d_model)
        #self.cross_attn = MultiHeadAttention(num_heads=num_heads,dim_in=d_in,d_model=d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        #self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, tgt_mask = None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        #attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        #x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        max_l: int,
        dim_in: int = 512,
        dim_model: int = 512,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(dim_in, dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model=dim_model,max_len=max_l)
        self.embedding = nn.Embedding(vocab_size, dim_in)

    def forward(self, tgt: Tensor, y=None, return_map = False) -> Tensor:
        if (not return_map) and self.training:
            m2 = generate_tgt_mask(tgt)
            tgt = self.positional_encoding(self.embedding(tgt))       #self.embedding(tgt)  torch.Size([16, 32, 64])
            for layer in self.layers:
                tgt = layer(tgt, m2)

            return self.linear(tgt)
        elif return_map:
            m2 = generate_tgt_mask(tgt)
            tgt = self.positional_encoding(self.embedding(tgt))
            for layer in self.layers:
                tgt = layer(tgt, m2)
            att_maps = []
            for L in self.layers:
                att_maps.append(L.self_attn.heads[-1].atten_map)
            return self.linear(tgt),att_maps
            
        else:
            assert y is not None
            m2 = generate_tgt_mask(tgt)
            tgt = self.positional_encoding(self.embedding(tgt))
            for layer in self.layers:
                tgt = layer(tgt, m2)
            pred = self.linear(tgt)
            loss_f = torch.nn.CrossEntropyLoss()
            loss = loss_f(pred.transpose(2,1), y)
            return loss
            


class Classifier(nn.Module):
    def __init__(self, numLayer, maxL,num_heads, input_size, dim_model, hidden_size, dim_ff,num_classes,rela_P=False):
        super(Classifier, self).__init__()
        self.transformerencoder = TransformerEncoder(
            num_layers=numLayer,
            max_l=maxL,
            dim_in=input_size,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_ff,
            rela_P=rela_P
        )
        self.rela = rela_P
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_map = False ):
        if not return_map:
            x = self.transformerencoder(x)
            x = torch.mean(x, dim=1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
        else:
            x = self.transformerencoder(x)
            x = torch.mean(x, dim=1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

            att_maps = []
            for L in self.transformerencoder.layers:
                if self.rela:
                    att_maps.append(L.self_attn.atten_map[:,0,:,:])
                else:
                    att_maps.append(L.self_attn.heads[0].atten_map)
        
            return x, att_maps


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, batch_size=16):
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.batch_size = batch_size

        self.head_dim = d_model//n_heads

        self.linears = _get_clones(nn.Linear(self.d_model, self.d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.relative_position_k = RelativePosition(self.head_dim, max_relative_position=16)
        self.relative_position_v = RelativePosition(self.head_dim, max_relative_position=16)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        self.atten_map = None

    def forward(self, query, key, value, mask):
        # embedding
        # query, key, value = [batch_size, len, hid_dim]
        batch_size = query.shape[0]
        query, key, value = [l(x).view(batch_size, -1, self.d_model) for l, x in
                             zip(self.linears, (query, key, value))]

        len_k = query.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        # Self-Attention
        # r_q1, r_k1 = [batch_size, len, n_heads, head_dim]
        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale.to(attn2.device)
        

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf')) #([32, 4, 10, 10])  [32, 1, 10]
        attn = torch.softmax(attn, dim=-1)
        self.atten_map = attn
        attn = self.dropout(attn)
        # attn = [batch_size, n_heads, len, len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size,len_q, -1)

        return self.linears[-1](x)




def _get_clones(module, N):
    return nn.ModuleList([module for _ in range(N)])

# torch.cuda.empty_cache()
# model = TransformerEncoder(1,100,rela_P=True)
# x = torch.randint(1,10,(32,10))
# print(model(x).shape)

