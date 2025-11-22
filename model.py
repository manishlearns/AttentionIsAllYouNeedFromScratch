import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(x):
        return self.embedding(x) * math.sqrt(self.d_model) # this is how its implemented in the research paper: sqrt(d)*embeddings 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model)

        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) # vector shape (seq_len,1)

        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1,seq_len, d_model) # for multiple sentences

        self.register_buffer('pe',pe) # these is not learned parameters, but stored as buffer, so that it will be present in state of the model

    def forward(x):
        x =  x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self,eps: float = 1e-05):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1)) # multiply
        self.beta = nn.Parameter(torch.zeros(1)) # added

    def forward(x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return (self.gamma * (x - mean) / (std + self.eps) ) + self.beta

class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model: int, d_ff: 2048, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Linear(in_features=d_model,out_features=d_ff)
  
        self.layer2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(x):

        # input: (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)

        x = self.layer1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

        # or return self.layer2(self.dropout(torch.relu(self.layer1(x))))

class MultiHeadAttentionNlock(nn.Module):
    def __init__(self,d_model:int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model must be divisible by h"       
        self.d_k = d_model // h

        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model)

    @staticmethod
    def attention(self,query,value,key,mask):

        # (batch,h, seq_len,d_k) * (batch,h, d_k, seq_len) --> (batch,h, seq_len,seq_len)
        scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(self.d_k) # (batch,h, seq_len,seq_len)

        if mask is not None:
            scores = scores.masked_fill_(mask==0,-1e9)

        attention = torch.softmax(scores,dim = -1) # (batch,h, seq_len,seq_len)

        if dropout is not None:
            attention_scores = dropout(attention)

        return (attention_scores@value), attention_scores


    def forward(self,q,v,k):

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        # now we need to split these matrices into multiple heads

        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2) # (batch,h, seq_len,d_k)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2) # (batch,h, seq_len,d_k)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2) # (batch,h, seq_len,d_k)

        x,self.attention_scores = MultiHeadAttentionNlock.attention(query,value,key,mask)

        # (batch,seq_len,h,d_k) --> (batch,h,seq_len,d_k) --> (batch,seq_len,d_model)

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.d_k*self.h) # (batch,seq_len,d_model)

        x = self.w_o(x)

        return x

    class ResidualConnection(nn.Module):
        def __init__(self,dropout: float):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization()

        def forward(self,x,sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

    class EncoderBlock(nn.Module):
        def __init__(self,self_attention: MultiHeadAttentionNlock, feed_forward: FeedForwardNetwork, dropout: float):
            super().__init__()
            self.self_attention = self_attention
            self.feed_forward = feed_forward
            self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

        def forward(self,x,src_mask):

            x = self.residual_connections[0](x,sublayer = lambda x : self.self_attention(x,x,x,src_mask))
            x = self.residual_connections[1](x,sublayer = self.feed_forward(x))

            return x

    class Encoder(nn.Module):
        def __init__(self, encoder_blocks: nn.ModuleList):
            super().__init__()
            self.encoder_blocks = encoder_blocks
            self.norm = LayerNormalization()

        def forward(self,x,src_mask):
            for block in self.encoder_blocks:
                x = block(x,src_mask)
            return self.norm(x)


    class DecoderBlock(nn.Module):
        def __init__(self, self_attention_block: MultiHeadAttentionNlock, cross_attention_block: MultiHeadAttentionNlock, feed_forward: FeedForwardNetwork, dropout: float):
            super().__init__()
            self.self_attention_block = self_attention_block
            self.cross_attention_block = cross_attention_block
            self.feed_forward = feed_forward
            self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

        def forward(self,x,encoder_output,src_mask,target_mask): # encoder mask(src_mask is used to avoid padding visibility in encoder)
            x  = self.residual_connections[0](x,sublayer = lambda x : self.self_attention_block(x,x,x,target_mask)) # target mask is the mask for decoder

            x = self.residual_connections[1](x,sublayer = lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
            x = self.residual_connections[2](x,sublayer = self.feed_forward(x))
            return x

    class Decoder(nn.Module):
        def __init__(self, decoder_blocks: nn.ModuleList):
            super().__init__()
            self.decoder_blocks = decoder_blocks
            self.norm = LayerNormalization()

        def forward(self,x,encoder_output,src_mask,target_mask):
            for block in self.decoder_blocks:
                x = block(x,encoder_output,src_mask,target_mask)
            return self.norm(x)

    class ProjectionLayer(nn.Module):
        def __init__(self,d_model: int, vocab_size: int):
            super().__init__()
            self.layer = nn.Linear(in_features=d_model, out_features=vocab_size)

        def forward(self,x):
            return torch.log_softmax(self.layer(x),dim = -1)
    
    class Transformer(mm.Module):
        def __init__(self,encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection: ProjectionLayer):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.src_embed = src_embed
            self.tgt_embed = tgt_embed
            self.src_pos = src_pos
            self.tgt_pos = tgt_pos
            self.projection = projection

        def encode(self,src,src_mask):
            src = self.src_embed(src)
            src = self.src_pos(src)
            return self.encoder(src,src_mask)

        def decode(self,encoder_output,src_mask,target,target_mask):
            target = self.tgt_embed(target)
            target = self.tgt_pos(target)
            return self.decoder(target,encoder_output,src_mask,target_mask)

        def project(self,x):
            return self.projection(x)

    def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, h: int = 8, N: int = 6, d_ff: int = 2048, dropout: float = 0.1 ) -> Transformer:

        # build embedding
        src_embed = InputEmbeddings(d_model,src_vocab_size)
        tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

        # postitional embeddings

        src_pos = PositionalEncoding(d_model, src_seq_len, dropout=dropout)
        tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout=dropout)

        # encoder blocks
        encoder_blocks = []

        for _ in range(N):

            encoder_self_attention = MultiHeadAttentionNlock(d_model,h,dropout=dropout)
            feed_forward = FeedForwardNetwork(d_model,d_ff,dropout=dropout)

            encoder_block = EncoderBlock(encoder_self_attention, feed_forward,dropout)
            encoder_blocks.append(encoder_block)

        # create decoder blocks
        
        decoder_blocks = []

        for _ in range(N):

            decoder_self_attention = MultiHeadAttentionNlock(d_model, h, dropout=dropout)
            decoder_cross_attention = MultiHeadAttentionNlock(d_model, h ,dropout=dropout)
            feed_forward = FeedForwardNetwork(d_model,d_ff, dropout=dropout)

            decoder_block = DecoderBlock(decoder_self_attention,decoder_cross_attention,feed_forward,dropout)

            decoder_blocks.append(decoder_block)

        
        # create encoder and decoder

        encoder = Encoder(encoder_blocks=encoder_blocks)
        decoder = Decoder(decoder_blocks=decoder_blocks)

        # projection

        projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

        # create transformer

        transformer = transformer(encoder,decoder, src_embed, tgt_embed, src_pos,tgt_pos,projection_layer)

        # initialize parameters:

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer


        
